#!/bin/bash

python src/infer.py --config exp/mobilenet_pretrain/config.yaml \
    --ckpt ckpt/mobilenet_pretrain/config/epochs_11.pth \
    --frame_scp /DKUdata/tangbl/courses/CS302_CV/final_project/src/frame_40191.scp