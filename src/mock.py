import socket
import json

def my_recv_utf8(s: socket.socket, total: int) -> str:
    bytes_recd = 0
    tmp = 0
    data = []
    while bytes_recd < total:
        tmp = s.recv(total - bytes_recd)
        if not tmp:
            return None
        bytes_recd = bytes_recd + len(tmp)
        data.append(tmp)
    return (b''.join(data)).decode(encoding='UTF-8')

def my_get_data(s: socket.socket):
    length = my_recv_utf8(s, 6) # nnnnnn
    if not length:
        return None
    data = my_recv_utf8(s, int(length))
    if not data:
        return None
    return json.loads(data)

if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_addr = ('0.0.0.0', 8888)
    server.bind(server_addr)

    print('start server')
    server.listen(1)

    while True:
        client, caddr = server.accept()
        print('connected {}'.format(caddr))
        # get message
        while True:
            message = my_get_data(client)
            if not message: # connection close
                print('Connection closed')
                break
            print(message)
            move = message['move']
            speed = message['speed']
            angle = message['angle']
            speed = speed / 1.5
