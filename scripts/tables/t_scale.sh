#!/bin/bash
export $(cat /home/ando/fm-playground/.env)
export PYTHONPATH='.'
cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py"

fastdevrun=no
exp_base_name=t_scale.1
overwrite=True


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


bsz_benv2=512

#train percent
corine_train_percent=1.0
benv2_train_percent=0.05 #same size as corine ~ 8000 samples
eurosat_train_percent=1.0

all_tasks=(
    
    #Resisc45 scale knn 12.5%: 0-5
    "base/dinov2 knn resisc45 800 1.0 0.125"
    "base/anysat_spot knn resisc45 800 1.0 0.125"
    "base/dofa knn resisc45 800 1.0 0.125"
    "base/senpamae knn resisc45 800 1.0 0.125"
    "base/panopticon_v2 knn resisc45 700 1.0 0.125" 
    "base/panopticon_v3 knn resisc45 700 1.0 0.125"

    #Resisc45 scale knn 25%: 6-11
    "base/dinov2 knn resisc45 800 1.0 0.25"
    "base/anysat_spot knn resisc45 800 1.0 0.25"
    "base/dofa knn resisc45 800 1.0 0.25"
    "base/senpamae knn resisc45 800 1.0 0.25"
    "base/panopticon_v2 knn resisc45 700 1.0 0.25"
    "base/panopticon_v3 knn resisc45 700 1.0 0.25"

    #Resisc45 scale knn 50%: 12-17
    "base/dinov2 knn resisc45 800 1.0 0.5"
    "base/anysat_spot knn resisc45 800 1.0 0.5"
    "base/dofa knn resisc45 800 1.0 0.5"
    "base/senpamae knn resisc45 800 1.0 0.5"
    "base/panopticon_v2 knn resisc45 700 1.0 0.5"
    "base/panopticon_v3 knn resisc45 700 1.0 0.5"

    #Resisc45 scale knn 100%: 18-23
    "base/dinov2 knn resisc45 800 1.0 1.0"
    "base/anysat_spot knn resisc45 800 1.0 1.0"
    "base/dofa knn resisc45 800 1.0 1.0"
    "base/senpamae knn resisc45 800 1.0 1.0"
    "base/panopticon_v2 knn resisc45 700 1.0 1.0"
    "base/panopticon_v3 knn resisc45 700 1.0 1.0"

    #Eurosat scale knn 12.5%: 24-31
    "base/croma_s2 knn geobench_eurosat_12b 800 1.0 0.125"
    "base/softcon_13b knn geobench_eurosat_13b 900 1.0 0.125"
    "base/anysat_s2 knn geobench_eurosat_10b 100 1.0 0.125"
    "base/galileo_s2_64 knn geobench_eurosat_10b 1000 1.0 0.125"
    "base/dofa knn geobench_eurosat_12b 800 1.0 0.125"
    "base/senpamae knn geobench_eurosat_12b 800 1.0 0.125"
    "base/panopticon_v2 knn geobench_eurosat_12b 500 1.0 0.125"
    "base/panopticon_v3 knn geobench_eurosat_12b 500 1.0 0.125"

    #Eurosat scale knn 25%: 32-39
    "base/croma_s2 knn geobench_eurosat_12b 800 1.0 0.25"
    "base/softcon_13b knn geobench_eurosat_13b 900 1.0 0.25"
    "base/anysat_s2 knn geobench_eurosat_10b 100 1.0 0.25"
    "base/galileo_s2_64 knn geobench_eurosat_10b 1000 1.0 0.25"
    "base/dofa knn geobench_eurosat_12b 800 1.0 0.25"
    "base/senpamae knn geobench_eurosat_12b 800 1.0 0.25"
    "base/panopticon_v2 knn geobench_eurosat_12b 500 1.0 0.25"
    "base/panopticon_v3 knn geobench_eurosat_12b 500 1.0 0.25"

    #Eurosat scale knn 50%: 40-47
    "base/croma_s2 knn geobench_eurosat_12b 800 1.0 0.5"
    "base/softcon_13b knn geobench_eurosat_13b 900 1.0 0.5"
    "base/anysat_s2 knn geobench_eurosat_10b 100 1.0 0.5"
    "base/galileo_s2_64 knn geobench_eurosat_10b 1000 1.0 0.5"
    "base/dofa knn geobench_eurosat_12b 800 1.0 0.5"
    "base/senpamae knn geobench_eurosat_12b 800 1.0 0.5"
    "base/panopticon_v2 knn geobench_eurosat_12b 500 1.0 0.5"
    "base/panopticon_v3 knn geobench_eurosat_12b 500 1.0 0.5"

    #Eurosat scale knn 100%: 48-55
    "base/croma_s2 knn geobench_eurosat_12b 800 1.0 1.0"
    "base/softcon_13b knn geobench_eurosat_13b 900 1.0 1.0" 
    "base/anysat_s2 knn geobench_eurosat_10b 100 1.0 1.0"
    "base/galileo_s2_64 knn geobench_eurosat_10b 1000 1.0 1.0"
    "base/dofa knn geobench_eurosat_12b 800 1.0 1.0"
    "base/senpamae knn geobench_eurosat_12b 800 1.0 1.0"
    "base/panopticon_v2 knn geobench_eurosat_12b 500 1.0 1.0"
    "base/panopticon_v3 knn geobench_eurosat_12b 500 1.0 1.0"
    
    #brick_kiln scale knn 12.5%: 56-63
    "base/croma_s2 knn geobench_brick_kiln_12b 800 1.0 0.125"
    "base/softcon_13b knn geobench_brick_kiln_13b 900 1.0 0.125"
    "base/anysat_s2 knn geobench_brick_kiln_10b 100 1.0 0.125"
    "base/galileo_s2_64 knn geobench_brick_kiln_10b 1000 1.0 0.125"
    "base/dofa knn geobench_brick_kiln_12b 800 1.0 0.125"
    "base/senpamae knn geobench_brick_kiln_12b 800 1.0 0.125"
    "base/panopticon_v2 knn geobench_brick_kiln_12b 500 1.0 0.125"
    "base/panopticon_v3 knn geobench_brick_kiln_12b 500 1.0 0.125"

    #brick_kiln scale knn 25%: 64-71
    "base/croma_s2 knn geobench_brick_kiln_12b 800 1.0 0.25"
    "base/softcon_13b knn geobench_brick_kiln_13b 900 1.0 0.25"
    "base/anysat_s2 knn geobench_brick_kiln_10b 100 1.0 0.25"
    "base/galileo_s2_64 knn geobench_brick_kiln_10b 1000 1.0 0.25"
    "base/dofa knn geobench_brick_kiln_12b 800 1.0 0.25"
    "base/senpamae knn geobench_brick_kiln_12b 800 1.0 0.25"
    "base/panopticon_v2 knn geobench_brick_kiln_12b 500 1.0 0.25"
    "base/panopticon_v3 knn geobench_brick_kiln_12b 500 1.0 0.25"

    #brick_kiln scale knn 50%: 72-79
    "base/croma_s2 knn geobench_brick_kiln_12b 800 1.0 0.5"
    "base/softcon_13b knn geobench_brick_kiln_13b 900 1.0 0.5"
    "base/anysat_s2 knn geobench_brick_kiln_10b 100 1.0 0.5"
    "base/galileo_s2_64 knn geobench_brick_kiln_10b 1000 1.0 0.5"
    "base/dofa knn geobench_brick_kiln_12b 800 1.0 0.5"
    "base/senpamae knn geobench_brick_kiln_12b 800 1.0 0.5"
    "base/panopticon_v2 knn geobench_brick_kiln_12b 500 1.0 0.5"
    "base/panopticon_v3 knn geobench_brick_kiln_12b 500 1.0 0.5"

    #brick_kiln scale knn 100%: 80-87
    "base/croma_s2 knn geobench_brick_kiln_12b 800 1.0 1.0"
    "base/softcon_13b knn geobench_brick_kiln_13b 900 1.0 1.0"
    "base/anysat_s2 knn geobench_brick_kiln_10b 100 1.0 1.0"
    "base/galileo_s2_64 knn geobench_brick_kiln_10b 1000 1.0 1.0"
    "base/dofa knn geobench_brick_kiln_12b 800 1.0 1.0"
    "base/senpamae knn geobench_brick_kiln_12b 800 1.0 1.0"
    "base/panopticon_v2 knn geobench_brick_kiln_12b 500 1.0 1.0"
    "base/panopticon_v3 knn geobench_brick_kiln_12b 500 1.0 1.0"

    #pv4ger_cls scale knn 12.5%: 88-93
    "base/dinov2 knn geobench_pv4ger_cls 1000 1.0 0.125"
    "base/anysat_spot knn geobench_pv4ger_cls 500 1.0 0.125"
    "base/dofa knn geobench_pv4ger_cls 1000 1.0 0.125"
    "base/senpamae knn geobench_pv4ger_cls 800 1.0 0.125"
    "base/panopticon_v2 knn geobench_pv4ger_cls 700 1.0 0.125" 
    "base/panopticon_v3 knn geobench_pv4ger_cls 700 1.0 0.125"

    #pv4ger_cls scale knn 25%: 94-99
    "base/dinov2 knn geobench_pv4ger_cls 1000 1.0 0.25"
    "base/anysat_spot knn geobench_pv4ger_cls 500 1.0 0.25"
    "base/dofa knn geobench_pv4ger_cls 1000 1.0 0.25"
    "base/senpamae knn geobench_pv4ger_cls 800 1.0 0.25"
    "base/panopticon_v2 knn geobench_pv4ger_cls 700 1.0 0.25" 
    "base/panopticon_v3 knn geobench_pv4ger_cls 700 1.0 0.25"

    #pv4ger_cls scale knn 50%: 100-105
    "base/dinov2 knn geobench_pv4ger_cls 1000 1.0 0.5"
    "base/anysat_spot knn geobench_pv4ger_cls 500 1.0 0.5"
    "base/dofa knn geobench_pv4ger_cls 1000 1.0 0.5"
    "base/senpamae knn geobench_pv4ger_cls 800 1.0 0.5"
    "base/panopticon_v2 knn geobench_pv4ger_cls 700 1.0 0.5" 
    "base/panopticon_v3 knn geobench_pv4ger_cls 700 1.0 0.5"

    #pv4ger_cls scale knn 100%: 106-111
    "base/dinov2 knn geobench_pv4ger_cls 1000 1.0 1.0"
    "base/anysat_spot knn geobench_pv4ger_cls 500 1.0 1.0"
    "base/dofa knn geobench_pv4ger_cls 1000 1.0 1.0"
    "base/senpamae knn geobench_pv4ger_cls 800 1.0 1.0"
    "base/panopticon_v2 knn geobench_pv4ger_cls 700 1.0 1.0" 
    "base/panopticon_v3 knn geobench_pv4ger_cls 700 1.0 1.0"
    
    
)

########## linear probe defaults

optim=sgd

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both

epochs=50
num_workers=4
check_val_every_n_epoch=10
val_subset=-1
nb_knn="[1,3,5,10,15,20,25,30]"

export CUDA_VISIBLE_DEVICES=$cuda_device

# Loop through each task ID provided
for task_id in "${task_ids[@]}"; do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3
    batch_size=$4
    train_subset=$5
    scale=$6

    cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$exp_base_name/$ds/$model/$training_mode/ \
        +model.training_mode=$training_mode \
        ++batch_size=$batch_size \
        num_workers=$num_workers \
        num_gpus=1 \
        seed=21 \
        ++overwrite=$overwrite \
        "

    if [ $fastdevrun == 'no' ]; then
        cmd="$cmd epochs=$epochs batch_size=$batch_size trainer.check_val_every_n_epoch=$check_val_every_n_epoch dataset.subset.train=$train_subset"
        # dataset.subset.val=$train_subset"

    elif [ $fastdevrun == 'fast' ]; then
        echo "fastdevrun 'fast'!"
        cmd="$cmd epochs=1 batch_size=32 trainer.check_val_every_n_epoch=1"
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"
        lrs_partial_ft="0.1"

    elif [ $fastdevrun == 'bsz' ]; then
        echo "fastdevrun 'bsz'!"
        cmd="$cmd epochs=1 batch_size=$batch_size trainer.check_val_every_n_epoch=1"
        s=$((batch_size * 1))
        echo $s
        cmd="$cmd dataset.subset.train=$s dataset.subset.val=$s dataset.subset.test=$s"
        lrs_partial_ft="0.1"

    else
        echo "fastdevrun not recognized"
        exit 1
    fi

    echo -e "\n\n**************** Running Task: ****************\n\n"
    echo "Task ID: $task_id"
    echo "Task: $task"
    echo "Model: $model"
    echo "Training Mode: $training_mode"
    echo "Dataset: $ds"
    echo "Batch Size: $batch_size"
    echo "Output Dir: $output_dir"
    echo "Check Val Every N Epoch: $check_val_every_n_epoch"
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    echo -e "\n\n************************************************\n\n"


    if [ $training_mode == 'linear_probe' ]; then
        
        cmd="$cmd \
            +optim=\${_optims.$optim} \
            "
        echo $cmd
        $cmd
            
    elif [ $training_mode == 'partial_finetune' ]; then

        for lr in $lrs_partial_ft; do
            echo "partial finetune with lr=$lr"
            lr_cmd="$cmd \
                +lr=$lr \
                +model.params_to_train=[] \
                warmup_epochs=$warmup_epochs \
                "
            echo $lr_cmd
            $lr_cmd
        done

    elif [ $training_mode == 'knn' ]; then

        cmd="$cmd \
        +temperature=$temperature \
        +nb_knn=$nb_knn \
        ++num_workers=3 \
        ++dataset.scale=$scale \
        "
        echo $cmd
        $cmd
    fi
            

    # collect results
    $PY_EXECUTABLE $REPO_PATH/geofm_src/collect_results.py $ODIR/$exp_base_name/

done

#Run like this:
# ./t2.sh 0 1 2  # Run tasks 0, 1, and 2
# ./t2.sh {0..3}  # Run tasks 0 through 3


