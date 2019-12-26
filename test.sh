#!/bin/bash

#Fill in the arguments as you choose.
#To run this script:
#  run 'chmod +x test.sh' in bash
#  run './test.sh'

python eval.py --checkpoint ./ckpt/diceLoss_patch256_aug-default-ngpus1-batchSize4-lr0.001-epoch250best_model_166.pth\
               --id diceLoss_patch256_aug-default-ngpus1-batchSize4-lr0.001-epoch250best_model_166_new\
	       --patch_size 256\
               --img_size 512\
              --batch_size 4\
               --visualize
