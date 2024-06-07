#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python ../main_plcc.py \
    -a resnet50 \
    --threshold-neg 0.9 --threshold-pos 0.9 --temperature 0.6\
    --num-classes 4 --batch-size 128 --dataset 'her2' --checkpoint './checkpoint-her2'\
    --data-path 'train.txt' \
    --dist-url 'tcp://localhost:10002' --multiprocessing-distributed \
    --world-size 1 --rank 0 \
    DATA_PATH/HER2 > ../log/her2_plcc_train.log 2>&1 &

