#!/bin/bash

python train.py \
    --in-size 320 576 \
    --checkpoint workspace/mot16-20210109-01-multiscale/jde.pth \
    --batch-size 64 \
    --scale-step 320 160 2 576 288 \
    --rescale-freq 80 \
    --workers 16 \
    --epochs 50 \
    --lr 0.025 \
    --milestones -1 -1 \
    --weight-decay 0.0001 \
    --savename jde \
    --pin \
    --workspace workspace/mot16-20210109-01-multiscale \
    --backbone shufflenetv2 \
    --thin 1.0x \
    --dataset-root /data/tseng/dataset/jde \
    --lr-coeff 1 1 1 \
    --box-loss diouloss \
    --cls-loss crossentropyloss
