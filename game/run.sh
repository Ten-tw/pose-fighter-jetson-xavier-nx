#!/bin/bash
xhost +
sudo docker run -it --rm \
    --runtime nvidia \
    --network host \
    --ipc=host \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/app \
    --device /dev/video0:/dev/video0 \
    pose-fighter:v1
