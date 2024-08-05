#!/bin/bash

python train_cnn.py \
    --task object \
    --explanation_type none \
    --model_type resnet50 \
    --num_epochs 3 \
    --transformation GRADIA \
    --batch_size 32 \
    --att_weight 0.3 \
    --exp_weight 1.5
