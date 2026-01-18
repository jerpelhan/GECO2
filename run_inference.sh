#!/bin/bash

conda activate cnt2

python inference.py \
--model_name GECO2FSCD \
--data_path /d/hpc/projects/FRI/pelhanj/fsc147 \
--model_path /d/hpc/projects/FRI/pelhanj/CNT_SAM2/models/ \
--backbone resnet50 \
--reduction 16 \
--image_size 1024 \
--emb_dim 256 \
--num_heads 8 \
--kernel_dim 1 \
--num_objects 3 \
--batch_size 2 \