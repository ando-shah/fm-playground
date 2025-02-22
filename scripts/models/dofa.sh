#!/bin/bash
export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'

######## classification
all_tasks=(
    'base/dofa linear_probe tropical_cyclone'
    'base/dofa linear_probe corine_21'
    'base/dofa linear_probe corine_sd'
    'base/dofa linear_probe fmow_8b'
    'base/dofa linear_probe fmow_4b'
    'base/dofa linear_probe geobench_eurosat_13b'
    'base/dofa linear_probe hyperview_21b'
    'base/dofa linear_probe digital_typhoon_3b'
    'base/dofa linear_probe digital_typhoon_1b'
    'base/dofa linear_probe geobench_eurosat_10b'
    )

suffix='debug'

for task in "${all_tasks[@]}"
do
    echo "Running Task: $task"

    # for lr in 0.002 0.005
    # do

    set -- $task
    model=$1
    training_mode=$2
    ds=$3
    #        lr=0.002 \

    $cmd \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$model/$training_mode/$ds/$suffix \
        +model.training_mode=$training_mode \
        \
        epochs=1 \
        warmup_epochs=0 \
        \
        batch_size=200 \
        num_workers=8 \
        num_gpus=1 \
        seed=13 \
        +trainer.fast_dev_run=True \
        \
        dataset.subset.train=0.1 # just as example how to use subset
            
    # done
done