#!/bin/bash

#Fill in the arguments as you choose.
#To run this script:
#  run 'chmod +x train.sh' in bash
#  run './train.sh'
python train.py --id hypoxia\
                --batch_size 32\
                --lr 3e-4\
                --num_epoch 250\
                --patch-size 128\
                --test-img '/home/haotian/Code/vessel_segmentation/data/hypoxia img/testing/M011 G2M5'
