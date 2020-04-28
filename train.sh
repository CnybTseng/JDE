#!/bin/bash

python train.py \
    --checkpoint workspace/caltech/jde.pth \
    --dataset dataset/caltech/ \
    --lr 0.01 \
    --milestones 15080,16965 \
    --pin \
    --workspace workspace/caltech/