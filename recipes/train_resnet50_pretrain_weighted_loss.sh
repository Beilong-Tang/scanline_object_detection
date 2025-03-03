#!/bin/bash

export CUDA_VISIBLE_DEVICES='4' && python src/train.py --config exp/resnet50_pretrain_weighted_loss/config.yaml