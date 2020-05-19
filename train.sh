#!/bin/bash

python train.py \
    --in-size 320 576 \
    --checkpoint workspace/mot16-2020-5-19/jde.pth \
    --dataset dataset/mot16/ \
    --scale-step 224 512 10 480 768 \
    --rescale-freq 99999999 \
    --workers 8 \
    --epochs 50 \
    --lr 0.001 \
    --milestones 16625 24937 \
    --weight-decay 0.0001 \
    --savename jde \
    --pin \
    --workspace workspace/mot16-2020-5-19/