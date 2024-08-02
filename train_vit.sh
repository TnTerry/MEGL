#!/bin/bash

python train_vit.py \
    --task object \
    --explanation_type visual \
    --num_epochs 3 \
    --transformation GRADIA \
    --batch_size 1 \
    --att_weight 0.3 \
    --exp_weight 1.5
