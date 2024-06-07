# PLCC

This is a PyTorch implementation of the paper "Partial-Label Contrastive Representation Learning for Fine-grained Biomarkers Prediction from Histopathology Whole Slide Images".

<!-- ```angular2html
@inproceedings{
}
``` -->


### Data Preparation

This code use "train.txt" to store the path and pseudo-label of images. An example of "train.txt" file is described as follows:

```angular2html
<path>                         <pseudo-label>
[path to slide1]/0000_0000.jpg 0
[path to slide1]/0000_0001.jpg 0
...
[path to slide2]/0000_0000.jpg 1
...
```

Note: we assign the pseudo-label for the patches from a WSI as the same of the WSI.

### Training

Use "default" contrastive table to train the model by following command. This mode will construct the negative sample pair from all other classes of lesion queue.

```angular2html
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python ../main_plcc.py \
    -a resnet50 \
    --threshold-neg 0.9 --threshold-pos 0.9 --temperature 0.6\
    --num-classes 4 --batch-size 128 --dataset 'her2' --checkpoint './checkpoint-her2'\
    --data-path 'train.txt' \
    --dist-url 'tcp://localhost:10002' --multiprocessing-distributed \
    --world-size 1 --rank 0 \
    DATA_PATH/HER2 > ../log/her2_plcc_train.log 2>&1 &
```
