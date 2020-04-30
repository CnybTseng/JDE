#!/bin/bash

python train.py \
    --checkpoint workspace/caltech/jde.pth \
    --dataset dataset/caltech/ \
    --interval 1 \
    --epochs 30 \
    --lr 0.01 \
    --milestones 5655,8483 \
    --weight-decay 0.0001 \
    --pin \
    --workspace workspace/caltech/