#!/bin/bash
export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'

######## classification
all_tasks=(
    'base/dofa linear_probe geobench_eurosat_10b'
    'base/dofa linear_probe geobench_eurosat_12b'
    'base/dofa linear_probe geobench_eurosat_13b'
    'base/dofa linear_probe geobench_eurosat_rgb'
    'base/dofa linear_probe geobench_brick_kiln_13b'
    'base/dofa linear_probe geobench_brick_kiln_12b'
    'base/dofa linear_probe geobench_brick_kiln_10b'
    'base/dofa linear_probe geobench_brick_kiln_rgb'
    'base/dofa linear_probe geobench_forest_6b'
    'base/dofa linear_probe geobench_forest_rgb'
    'base/dofa linear_probe geobench_neontree'
    'base/dofa linear_probe geobench_nzcattle'
    'base/dofa linear_probe geobench_pv4ger_cls'
    'base/dofa linear_probe geobench_pv4ger_seg'
    'base/dofa linear_probe geobench_sacrop_10b'
    'base/dofa linear_probe geobench_sacrop_rgb'
    'base/dofa linear_probe geobench_sacrop_12b'
    'base/dofa linear_probe geobench_so2sat_10b'
    'base/dofa linear_probe geobench_so2sat_rgb'
    'base/dofa linear_probe geobench_brick_kiln_13b'
    'base/dofa linear_probe geobench_brick_kiln_12b'
    'base/dofa linear_probe geobench_brick_kiln_rgb'
    'base/dofa linear_probe geobench_chesapeake_4b'
    'base/dofa linear_probe geobench_cashew_10b'
    'base/dofa linear_probe geobench_cashew_12b'
    'base/dofa linear_probe geobench_cashew_rgb'

    'base/dofa linear_probe resisc45'

    'base/dofa linear_probe benv2_s1'
    'base/dofa linear_probe benv2_s2_12b'
    'base/dofa linear_probe benv2_s2_10b'
    'base/dofa linear_probe benv2_rgb'


    'base/dofa linear_probe benv2_s2_4b'

    'base/dofa linear_probe tropical_cyclone'
    'base/dofa linear_probe corine_21'


    'base/dofa linear_probe corine_sd'
    'base/dofa linear_probe corine_s2_10b'
    'base/dofa linear_probe corine_10b'
    'base/dofa linear_probe corine_1b'

    'base/dofa linear_probe hyperview_10b'
    'base/dofa linear_probe hyperview_21b'

    'base/dofa linear_probe fmow_8b'
    'base/dofa linear_probe fmow_rgb'
    'base/dofa linear_probe fmow_4b'
    
    'base/dofa linear_probe digital_typhoon_1b'
    'base/dofa linear_probe digital_typhoon_3b'
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
        dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64 # just as example how to use subset
            
    # done
done