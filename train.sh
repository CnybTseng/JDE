#!/bin/bash

python train.py \
    --in-size 320 576 \
    --checkpoint workspace/task-2020-5-23/jde.pth \
    --dataset workspace/task-2020-5-23/ \
    --scale-step 224 512 10 480 768 \
    --rescale-freq 99999999 \
    --workers 8 \
    --epochs 50 \
    --lr 0.01 \
    --milestones 16625 24937 \
    --weight-decay 0.0001 \
    --savename jde \
    --pin \
    --workspace workspace/task-2020-5-23/