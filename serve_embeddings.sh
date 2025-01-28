#!/bin/bash

port=7997
model1=lightonai/modernbert-embed-large
volume=$PWD/data/infinity

docker run -it --gpus all \
 -v $volume:/app/.cache \
 -p $port:$port \
 michaelf34/infinity:latest \
 v2 \
 --device cuda \
 --model-id $model1 \
 --port $port