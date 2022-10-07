#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#depthwise_capsule True or False
#squash sabour (original one), soft or unit
#routing sabour(original one) or unit
#model name ucaps, onlycaps-3d or unet

python ../evaluate.py --gpus 1 \
                --model_name ucaps \
                --squash unit\
                --routing unit\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --gpus 1 \
                --save_image 0 \
                --model_name ucaps \
                --dataset task04_hippocampus \
                --fold 0 \
                --checkpoint_path /logs/ucaps_task04_hippocampus_0/version_4/checkpoints \
                --val_patch_size 32 32 32 \
                --sw_batch_size 1 \
                --overlap 0.75
                
python ../evaluate.py --gpus 1 \
                --model_name ucaps \
                --squash unit\
                --routing unit\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --gpus 1 \
                --save_image 0 \
                --model_name ucaps \
                --dataset task04_hippocampus \
                --fold 1 \
                --checkpoint_path /logs/ucaps_task04_hippocampus_1/version_4/checkpoints \
                --val_patch_size 32 32 32 \
                --sw_batch_size 1 \
                --overlap 0.75
                
python ../evaluate.py --gpus 1 \
                --model_name ucaps \
                --squash unit\
                --routing unit\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --gpus 1 \
                --save_image 0 \
                --model_name ucaps \
                --dataset task04_hippocampus \
                --fold 2 \
                --checkpoint_path /logs/ucaps_task04_hippocampus_2/version_4/checkpoints \
                --val_patch_size 32 32 32 \
                --sw_batch_size 1 \
                --overlap 0.75
                
python ../evaluate.py --gpus 1 \
                --model_name ucaps \
                --squash unit\
                --routing unit\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --gpus 1 \
                --save_image 0 \
                --model_name ucaps \
                --dataset task04_hippocampus \
                --fold 3 \
                --checkpoint_path /logs/ucaps_task04_hippocampus_3/version_4/checkpoints \
                --val_patch_size 32 32 32 \
                --sw_batch_size 1 \
                --overlap 0.75
