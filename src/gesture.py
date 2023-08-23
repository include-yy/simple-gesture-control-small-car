import socket

import mediapipe as mp
import cv2
import numpy as np
from scipy.ndimage import median_filter
import math
import time
import json

# pv: projection direction
# get speed from projection direction
# return value betweens 0 to 1, float
def speed_control(pv):
    a, b, c = pv
    d = math.sqrt(a*a + b*b)
    e = math.sqrt(c*c + d*d)
    speed = d / e
    return round(speed, 1)


# Calculate the angle between the projection of the vector
# on the XY plane and the X-axis.
# Result between 0 and 360 degree
def calculate_angle(pv):
    angle = math.atan2(pv[1], pv[0])
    if angle < 0:
        angle = angle + math.pi * 2
    return math.degrees(angle)


# image filtering process
def median_filter_same_size(image, kernel_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = median_filter(gray_image, size=kernel_size, mode='mirror')
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR)
    return filtered_image


# get normal vector
def get_projection_vector(hand21):
    matrix = np.zeros((3, 21))
    for j in range(21):
        matrix[0][j] = hand21[j].x
        matrix[1][j] = hand21[j].y
        matrix[2][j] = hand21[j].z

    # get speed and angle from gesture
    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    result = np.dot(rotation_matrix, matrix)
    result = result[:, [0, 5, 9]]
    ab = np.array([result[0][1] - result[0][0],
                   result[1][1] - result[1][0],
                   result[2][1] - result[2][0]])
    ac = np.array([result[0][2] - result[0][0],
                   result[1][2] - result[1][0],
                   result[2][2] - result[2][0]])
    normal = [ab[1] * ac[2] - ab[2] * ac[1],
              ab[2] * ac[0] - ab[0] * ac[2],
              ab[0] * ac[1] - ab[1] * ac[0]]
    return normal[0], normal[1], -normal[2]


def get_vec_angle_360(v1, v2):
    g1 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]
    g2 = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
    cos_val = dot / (math.sqrt(g1) * math.sqrt(g2))
    return math.acos(cos_val) * 180 / math.pi


def get_angle_three_point(vs):
    v1 = (vs[1].x-vs[0].x, vs[1].y-vs[0].y, vs[1].z-vs[0].z)
    v2 = (vs[2].x-vs[1].x, vs[2].y-vs[1].y, vs[2].z-vs[1].z)
    return get_vec_angle_360(v1, v2)


# get fist gesture
def get_gesture(hand):
    fingers1 = hand[5:8]
    fingers2 = hand[9:12]
    fingers3 = hand[13:16]
    fingers4 = hand[17:20]
    cnt = 0
    for i in (fingers1, fingers2, fingers3, fingers4):
        if get_angle_three_point(i) < 80:
            cnt = cnt + 1

    return True if cnt >= 1 else False


def main():
    previous_angle = 0

    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # soc.connect(('192.168.126.128', 8888))
    soc.connect(('127.0.0.1', 8888))
    # open camera
    cap = cv2.VideoCapture(0)
    # hand check object

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils

    while True:
        success, img = cap.read()
        if not success:
            continue
        # process hand image
        img = cv2.flip(img, 1)
        height, width, _ = np.shape(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # hands detected
        if results.multi_hand_landmarks:
            # get the 0th hand in hands list
            hand_21 = results.multi_hand_landmarks[0]
            landmark = []
            for i in range(21):
                landmark.append(hand_21.landmark[i])

            projection_direction = get_projection_vector(landmark)
            print(projection_direction)
            speed = speed_control(projection_direction)
            angle = int(calculate_angle(projection_direction))

            # if delta angle is too small, dismiss it
            if abs(angle - previous_angle) > 5:
                previous_angle = angle
            # if z value is negative, angle also need flip
            if projection_direction[2] <= 0:
                angle = -angle
            # angle = str(angle)

            gesture = get_gesture(landmark)

            # gesture = get_str_gesture(up_fingers)
            # print(str_gesture)
            # send json data
            motion = {'move': gesture,
                      'speed': speed,
                      'angle': angle}
            data = json.dumps(motion)
            print(data)
            stream = data.encode(encoding='UTF-8')
            length = '{:0>6d}'.format(len(stream)).encode(encoding='UTF-8')
            soc.sendall(b''.join([length, stream]))
            # draw keypoint
            list_lms = []
            for i in range(21):
                pos_x = hand_21.landmark[i].x * width
                pos_y = hand_21.landmark[i].y * height
                list_lms.append([int(pos_x), int(pos_y)])
            mp_draw.draw_landmarks(img, hand_21, mp_hands.HAND_CONNECTIONS)
            time.sleep(0.01)
        cv2.imshow("hands", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
