#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

bash ./train_ucaps_hippocampus_unitunit.sh
bash ./train_ucaps_hippocampus_softunit.sh
bash ./train_ucaps_hippocampus.sh
bash ./train_ucaps_hippocampus_depth.sh
