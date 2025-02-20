#!/bin/bash
echo "Contents of the current directory:"
ls -lah

export CUDA_VISIBLE_DEVICES=0
export GEO_BENCH_DIR=/mnt/data/cc_benchmark
export MODEL_WEIGHTS_DIR=/mnt/data/fm_weights
export PYTHONPATH="."
export HYDRA_FULL_ERROR=1
export ODIR="."


model="galileo"
dataset="geobench_eurosat_10b"

training_mode=linear_probe
params_to_train="[model.projector_s2.patch_embed.inconv.0.weight,\
model.projector_s2.patch_embed.inconv.0.bias]"

batch_size="64"
task="classification"
lr="0.002"
epochs="30"
warmup_epochs="3"
pretrained_path=/mnt/data/fm_weights/galileo/encoder.pt

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

/home/toolkit/.conda/envs/dofaEnv/bin/python geofm_src/main.py \
output_dir=/mnt/results/nils/exps/${model}_${dataset} \
training_mode=${training_mode} \
dataset=${dataset} \
batch_size=${batch_size} \
model=${model} \
+model.training_mode=${training_mode} \
+model.params_to_train=${params_to_train} \
+model.pretrained_path=${pretrained_path} \
dataset.input_key=s2 \
dataset.image_resolution=128 \
model.image_resolution=128 \
+model.patch_size=8 \
\
lr=${lr} \
epochs=${epochs} \
warmup_epochs=${warmup_epochs} \
\
+task=${task} \
num_workers=8 \
seed=42 \
num_gpus=${num_gpus} \
dataset.data_path=${data_path} \
+trainer.fast_dev_run=True