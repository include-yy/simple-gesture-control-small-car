# simple-gesture-control-car

The project employed the gesture recognition feature of MediaPipe to facilitate basic control over a small car through hand gestures.

## Requests

- camera
- LIMO robotic car

## Install

This project utilizes [pdm](https://pdm.fming.dev/latest/) for dependency management. By leveraging it, we can effortlessly establish virtual environments and lock package versions.

```
git clone https://github.com/include-yy/simple-gesture-control-small-car
cd simple-gesture-control-small-car
pdm install
```

## Test

```
pdm run src\mock.py
# in another terminal
pdm run src\gesture.py
```

If successful, you should see a window displaying the camera feed and observe the transmitted data in the terminal corresponding to mock.py.

## USE

First, run limo.py on LIMO. Subsequently, modify the IP address and port in gesture.py to match the address of LIMO. Then, execute the following on a machine equipped with a camera:

```
pdm run src\gesture.py
```
