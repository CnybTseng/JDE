#!/bin/bash
for ((i=1; i<=$1; i=i+1))
do 
    echo running test $i ...
    $(pwd)/../build/mot-test \
    $(pwd)/../build/mot.yaml \
    $(pwd)/../../data/MOT16-01/,$(pwd)/../../data/MOT16-03/,$(pwd)/../../data/MOT16-06/,$(pwd)/../../data/MOT16-07/ \
    4
done