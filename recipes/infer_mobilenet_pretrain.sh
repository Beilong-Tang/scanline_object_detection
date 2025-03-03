#!/bin/bash

# python src/infer.py --config exp/mobilenet_pretrain/config.yaml \
#     --ckpt ckpt/mobilenet_pretrain/config/epochs_11.pth \
#     --frame_scp /DKUdata/tangbl/courses/CS302_CV/final_project/src/frame_40192.scp

python src/infer.py --config exp/mobilenet_from_scratch_augmentation/config.yaml \
    --ckpt ckpt/mobilenet_from_scratch_augmentation/config/epochs_16.pth \
    --frame_scp /DKUdata/tangbl/courses/CS302_CV/final_project/src/frame_40192.scp