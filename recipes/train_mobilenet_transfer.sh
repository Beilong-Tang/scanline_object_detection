#!/bin/bash

export CUDA_VISIBLE_DEVICES='4' && python src/train.py --config exp/mobilenet_transfer/config.yaml