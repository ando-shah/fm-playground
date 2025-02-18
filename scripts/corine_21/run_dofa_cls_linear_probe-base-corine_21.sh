#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export $(cat /home/ando/fm-playground/.env)
export MODEL_SIZE=base

model=dofa_cls_linear_probe
dataset=corine_21
batch_size=512
lr=0.002
epochs=5
warmup_epochs=3
task=classification
num_gpus=$(nvidia-smi -L | wc -l)

echo "*************************************************"
echo "Output Directory": $ODIR
echo "Model Size": $MODEL_SIZE
echo "Batch Size": $batch_size
echo "Learning Rate": $lr
echo "Epochs": $epochs
echo "Warmup Epochs": $warmup_epochs
echo "Task": $task
echo "Number of GPUs": $num_gpus
echo "*************************************************"

python /home/ando/fm-playground/src/main.py \
output_dir=${ODIR}/exps/${model}_${dataset} \
model=${model} \
dataset=${dataset} \
lr=${lr} \
task=${task} \
num_gpus=${num_gpus} \
num_workers=8 \
epochs=${epochs} \
warmup_epochs=${warmup_epochs} \
seed=13 \
batch_size=100 \
