#!/bin/sh

./ffmpeg -i rtsp://192.168.1.103:8554/ch20 -b:v 1024k -vcodec copy -r 25 -y ch20.avi