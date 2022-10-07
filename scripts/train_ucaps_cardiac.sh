#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

#depthwise_capsule True or False
#squash sabour (original one), soft or unit
#routing sabour(original one) or unit
#model name ucaps, onlycaps-3d or unet
python ../train.py --log_dir ./logs \
                --gpus 1 \
                --accelerator ddp \
                --model_name ucaps \
                --squash soft\
                --routing unit\
                --depthwise_capsule 1\
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --fold 0 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

python ../train.py --log_dir ./logs \
                --gpus 1 \
                --accelerator ddp \
                --model_name ucaps \
                --squash soft\
                --routing unit\
                --depthwise_capsule 1\
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --fold 1 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

python ../train.py --log_dir ./logs \
                --gpus 1 \
                --accelerator ddp \
                --model_name ucaps \
                --squash soft\
                --routing unit\
                --depthwise_capsule 1\
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --fold 2 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

python ../train.py --log_dir ./logs \
                --gpus 8 \
                --accelerator ddp \
                --model_name ucaps \
                --squash soft\
                --routing unit\
                --depthwise_capsule 1\
                --check_val_every_n_epoch 50 \
                --max_epochs 10000 \
                --dataset task02_heart \
                --fold 3 \
                --cache_rate 1.0 \
                --train_patch_size 128 128 128 \
                --num_workers 64 \
                --batch_size 1 \
                --num_samples 1 \
                --in_channels 1 \
                --out_channels 2 \
                --val_patch_size 128 128 128 \
                --val_frequency 50 \
                --sw_batch_size 2 \
                --overlap 0.75

