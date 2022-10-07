#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

#depthwise_capsule True or False
#squash sabour (original one), soft or unit
#routing sabour(original one) or unit
#model name ucaps, onlycaps-3d or unet

python ../train.py --gpus 1 \
                --model_name ucaps \
                --squash sabour\
                --routing sabour\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --fold 0 \
                --share_weight 0 \
                --cache_rate 1.0 \
                --train_patch_size 32 32 32 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 4 \
                --in_channels 1 \
                --out_channels 3 \
                --val_patch_size 32 32 32 \
                --val_frequency 1 \
                --sw_batch_size 4 \
                --overlap 0.75


python ../train.py --gpus 1 \
                --model_name ucaps \
                --squash sabour\
                --routing sabour\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --fold 1 \
                --share_weight 0 \
                --cache_rate 1.0 \
                --train_patch_size 32 32 32 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 4 \
                --in_channels 1 \
                --out_channels 3 \
                --val_patch_size 32 32 32 \
                --val_frequency 1 \
                --sw_batch_size 4 \
                --overlap 0.75



python ../train.py --gpus 1 \
                --model_name ucaps \
                --squash sabour\
                --routing sabour\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --fold 2 \
                --share_weight 0 \
                --cache_rate 1.0 \
                --train_patch_size 32 32 32 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 4 \
                --in_channels 1 \
                --out_channels 3 \
                --val_patch_size 32 32 32 \
                --val_frequency 1 \
                --sw_batch_size 4 \
                --overlap 0.75


python ../train.py --gpus 1 \
                --model_name ucaps \
                --squash sabour\
                --routing sabour\
                --depthwise_capsule 1\
                --accelerator ddp \
                --check_val_every_n_epoch 1 \
                --max_epochs 2000 \
                --dataset task04_hippocampus \
                --fold 3 \
                --share_weight 0 \
                --cache_rate 1.0 \
                --train_patch_size 32 32 32 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 4 \
                --in_channels 1 \
                --out_channels 3 \
                --val_patch_size 32 32 32 \
                --val_frequency 1 \
                --sw_batch_size 4 \
                --overlap 0.75

