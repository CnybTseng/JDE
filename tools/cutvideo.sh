#!/bin/sh

# $1: input video
# $2: start time with format hh:mm:ss
# $3: duration with format hh:mm:ss, or -to which means end time
# $4: output video (.mkv, .mp4 report error)

ffmpeg -i $1 -ss $2 -t $3 -c copy $4