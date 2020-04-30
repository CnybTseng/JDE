#!/bin/sh

python ../../split_dataset.py --path \
/data/tseng/dataset/jde/MOT16/train/MOT16-02/img1,\
/data/tseng/dataset/jde/MOT16/train/MOT16-04/img1,\
/data/tseng/dataset/jde/MOT16/train/MOT16-05/img1,\
/data/tseng/dataset/jde/MOT16/train/MOT16-09/img1,\
/data/tseng/dataset/jde/MOT16/train/MOT16-10/img1,\
/data/tseng/dataset/jde/MOT16/train/MOT16-11/img1,\
/data/tseng/dataset/jde/MOT16/train/MOT16-13/img1 \
-tr 0.5