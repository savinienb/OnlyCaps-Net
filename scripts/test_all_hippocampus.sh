#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

bash ./test_ucaps_hippocampus_unitunit.sh
bash ./test_ucaps_hippocampus_softunit.sh
bash ./test_ucaps_hippocampus.sh
bash ./test_ucaps_hippocampus_depth.sh
