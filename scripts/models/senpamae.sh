#!/bin/bash
export $(cat /home/ando/fm-playground/.env)
export PYTHONPATH='.'
cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py"

fastdevrun=true
exp_base_name=senpamae_test


# Parse CUDA device number from command line (default to 0)
cuda_device=0
if [[ "$1" == "--device" || "$1" == "-d" ]]; then
    cuda_device="$2"
    shift 2  # Remove these two arguments from $@
fi

# If no arguments provided, run all tasks
if [ $# -eq 0 ]; then
    # Generate sequence from 0 to (number of tasks - 1)
    task_ids=($(seq 0 $((${#all_tasks[@]}-1))))
else
    task_ids=("$@")
fi

bsz_brick_kiln=100
bsz_eurosat=1024
bsz_pv4ger=512
bsz_so2sat=1024
bsz_benv2=512
bsz_forestnet=512


all_tasks=(
    
    # 'base/dofa linear_probe hyperview_10b 64'

    # #SenPaMae

    "base/senpamae_13b linear_probe geobench_eurosat_13b 512" #fix
    "base/senpamae_12b linear_probe geobench_eurosat_12b 512" 
    "base/senpamae_rgb linear_probe resisc45 512"
    "base/senpamae_12b linear_probe benv2_s2_12b 512" 
    "base/senpamae_6b linear_probe geobench_forestnet_6b 512"
    "base/senpamae_8b linear_probe fmow_8b 512"
    "base/senpamae_rgb linear_probe fmow_rgb 512"


    "base/senpamae_10b linear_probe benv2_s2_10b 512"
    "base/senpamae_6b linear_probe geobench_forestnet_6b 512"
    "base/senpamae_10b linear_probe geobench_brick_kiln_10b 512"
    "base/senpamae_rgb linear_probe linear_probe geobench_pv4ger_cls 512"
    'base/senpamae_rgb linear_probe fmow_rgb 512' 
    'base/senpamae_10b linear_probe geobench_so2sat_10b 512'
    'base/senpamae_10b linear_probe geobench_eurosat_10b 512'
    'base/senpamae_8b linear_probe corine_sd 512' 

    'base/anysat_s2 linear_probe geobench_eurosat_10b 128'
    'base/anysat_s2 linear_probe geobench_so2sat_10b 128'
    'base/anysat_spot linear_probe geobench_pv4ger_cls 128' #fail
    'base/anysat_s2 linear_probe geobench_brick_kiln_10b 128'
    'base/anysat_l7 linear_probe geobench_forestnet_6b 128' #fail
    'base/anysat_s2 linear_probe benv2_s2_10b 128'
    'base/anysat_s1-asc linear_probe benv2_s1 128' #fail

    # #DOFA
    # "base/dofa linear_probe benv2_s2_12b ${bsz_benv2}"
    # "base/dofa linear_probe geobench_forestnet_6b ${bsz_forestnet}"
    # "base/dofa linear_probe geobench_brick_kiln_12b ${bsz_brick_kiln}"
    # "base/dofa linear_probe geobench_pv4ger_cls ${bsz_pv4ger}"  # Fixed missing $ before {
    # "base/dofa linear_probe geobench_eurosat_12b ${bsz_eurosat}"
    # "base/dofa linear_probe geobench_so2sat_10b ${bsz_so2sat}"  # Fixed missing $ before {
)

########## linear probe defaults
n_last_blocks_list='[1,4]'
pooling='[avgpool,cls,default]'
lrs_linear_probe='[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.2,0.3,0.5,1,3,5,10,20]'

########## pe linear probe (=partial finetune) defaults
lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both
epochs=50
batch_size=500
num_workers=8
check_val_every_n_epoch=10

export CUDA_VISIBLE_DEVICES=$cuda_device

# export CUDA_LAUNCH_BLOCKING=1

# Loop through each task ID provided
for task_id in "${task_ids[@]}"; do
    task=${all_tasks[$task_id]}

    ########## extract task specific parameters
    set -- $task
    model=$1
    training_mode=$2
    ds=$3
    batch_size=$4

    ########## execution
    cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$exp_base_name/$ds/$model/$training_mode/ \
        +model.training_mode=$training_mode \
        \
        epochs=$epochs \
        \
        ++batch_size=$batch_size \
        num_workers=$num_workers \
        num_gpus=1 \
        seed=21 \
        "

    if $fastdevrun; then
        echo "fastdevrun!"
        cmd="$cmd epochs=1 trainer.check_val_every_n_epoch=1 overwrite=True"
    else
        cmd="$cmd epochs=$epochs batch_size=$batch_size trainer.check_val_every_n_epoch=$check_val_every_n_epoch"
    fi



    echo "\n\n**************** Running Task: ****************\n\n"
    echo "Task ID: $task_id"
    echo "Task: $task"
    echo "Model: $model"
    echo "Training Mode: $training_mode"
    echo "Dataset: $ds"
    echo "Batch Size: $batch_size"
    echo "Output Dir: $output_dir"
    echo "Check Val Every N Epoch: $check_val_every_n_epoch"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo "\n\n************************************************\n\n"

    if [ $training_mode == 'linear_probe' ]; then
        if $fastdevrun; then
            cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"
        fi
        
        cmd="$cmd \
            +n_last_blocks_list=$n_last_blocks_list \
            +pooling=$pooling \
            +lr=$lrs_linear_probe"
        echo $cmd
        $cmd
            
    elif [ $training_mode == 'partial_finetune' ]; then
        if $fastdevrun; then
            cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"
            lrs_partial_ft="0.1 0.01"
        fi

        for lr in $lrs_partial_ft; do
            echo "partial finetune with lr=$lr"
            lr_cmd="$cmd \
                +lr=$lr \
                +base_lr=-1 \
                +model.params_to_train=[] \
                warmup_epochs=$warmup_epochs \
                "
            echo $lr_cmd
            $lr_cmd
        done
    fi
done

#Run like this:
# ./t2.sh 0 1 2  # Run tasks 0, 1, and 2
# ./t2.sh {0..3}  # Run tasks 0 through 3
# ./t3.sh - Uses CUDA device 0 and runs all tasks
# ./t3.sh --device 1 - Uses CUDA device 1 and runs all tasks
# ./t3.sh -d 2 0 1 - Uses CUDA device 2 and runs tasks 0 and 1
# ./t3.sh 0 1 2 - Uses default CUDA device 0 and runs tasks 0, 1, and 2