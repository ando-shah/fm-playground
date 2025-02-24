#!/bin/bash
export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'

######## classification
all_tasks=(
    
    'base/dofa linear_probe geobench_eurosat_12b'
    'base/dofa linear_probe geobench_eurosat_13b'
    'base/dofa linear_probe geobench_eurosat_rgb'

    
    'base/dofa linear_probe geobench_forest_rgb'
    
    
    'base/dofa linear_probe geobench_pv4ger_cls'
    'base/dofa linear_probe geobench_pv4ger_seg'
    
    'base/dofa linear_probe geobench_sacrop_rgb'
    'base/dofa linear_probe geobench_sacrop_12b'
    
    'base/dofa linear_probe geobench_brick_kiln_13b'
    'base/dofa linear_probe geobench_brick_kiln_12b'
    'base/dofa linear_probe geobench_brick_kiln_rgb'
    
   
    'base/dofa linear_probe geobench_cashew_12b'
    'base/dofa linear_probe geobench_cashew_rgb'

    'base/anysat linear_probe benv2_s1' # TODO: specify which sensor for anysat - "s1-asc"
    'base/anysat linear_probe benv2_s2_4b' # TODO: specify which sensor for anysat - "NAIP"
    'base/anysat linear_probe benv2_s2_10b' # TODO: specify which sensor for anysat - "s2"

    'base/anysat linear_probe resisc45' #TODO: specify which sensor for anysat - "spot"

    'base/anysat linear_probe geobench_so2sat_10b' #TODO: specify which sensor for anysat - "s2"
    'base/anysat linear_probe geobench_brick_kiln_10b' #TODO: specify which sensor for anysat - "s2"
    'base/anysat linear_probe geobench_forest_6b' #TODO: specify which sensor for anysat - "l7"
    'base/anysat linear_probe geobench_eurosat_10b' #TODO: specify which sensor for anysat - "s2"
    'base/anysat linear_probe geobench_pv4ger_cls' #TODO: specify which sensor for anysat - "spot"

    'base/anysat linear_probe geobench_pv4ger_seg' #TODO: specify which sensor for anysat - "spot"
    'base/anysat linear_probe geobench_chesapeake_4b' #TODO: specify which sensor for anysat - "NAIP"/"Aerial" pick based on GSD | Check band ordering matches AnySat's
    'base/anysat linear_probe geobench_cashew_10b' #TODO: specify which sensor for anysat - "s2"
    'base/anysat linear_probe geobench_sacrop_10b' #TODO: specify which sensor for anysat - "s2"
    'base/anysat linear_probe geobench_nzcattle' #TODO: specify which sensor for anysat - "SPOT"
    'base/anysat linear_probe geobench_neontree' #TODO: specify which sensor for anysat - "spot"

    'base/anysat linear_probe fmow_4b' #TODO: specify which sensor for anysat - "NAIP"/"Aerial" pick based on GSD | Check band ordering matches AnySat's. fmow_4b band order is RGB,NIR

    'base/dofa linear_probe corine_10b' #TODO: specify which sensor for anysat - "s2"
    'base/dofa linear_probe corine_s2_10b' #TODO: specify which sensor for anysat - "s2"

    'base/dofa linear_probe hyperview_10b' #TODO: apply a center crop to the image to match the input size of the model
    'base/dofa linear_probe digital_typhoon_3b' #TODO: apply a center crop to the image to match the input size of the model

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