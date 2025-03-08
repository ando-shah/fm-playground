#!/bin/bash
export $(cat /home/ando/fm-playground/.env)
export PYTHONPATH='.'
cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py"

fastdevrun=no
# exp_base_name=cross_sens_test
exp_base_name=cross_sens/fmow
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


#train percent
train_perc=0.1

all_tasks=(

    #A: 8b:4b
    # Baselines: 8b -> 0 to 2
    "base/panopticon_v3 knn fmow_8b fmow_8b 350 ${train_perc}"
    "base/dofa knn fmow_8b fmow_8b 1000 ${train_perc}"
    "base/senpamae knn fmow_8b fmow_8b 300 ${train_perc}"

    # Baselines: 4b -> 3 to 5
    "base/panopticon_v3 knn fmow_4b fmow_4b 350 ${train_perc}"
    "base/dofa knn fmow_4b fmow_4b 1000 ${train_perc}"
    "base/senpamae knn fmow_4b fmow_4b 300 ${train_perc}"

    # Baselines: s2 -> 6 to 8
    "base/panopticon_v3 knn fmow_s2 fmow_s2 350 ${train_perc}"
    "base/dofa knn fmow_s2 fmow_s2 1000 ${train_perc}"
    "base/senpamae knn fmow_s2 fmow_s2 300 ${train_perc}"

    #A: 8b:4b -> 9 to 11
    "base/panopticon_v3 knn fmow_8b fmow_4b 400 ${train_perc}"
    "base/dofa knn fmow_8b fmow_4b 1000 ${train_perc}"
    "base/senpamae knn fmow_8b fmow_4b 400 ${train_perc}"

    #B: 4b:8b -> 12 to 14
    "base/panopticon_v3 knn fmow_4b fmow_8b 400 ${train_perc}"
    "base/dofa knn fmow_4b fmow_8b 1000 ${train_perc}"
    "base/senpamae knn fmow_4b fmow_8b 400 ${train_perc}"

    #C: 8b:s2 -> 15 to 17
    "base/panopticon_v3 knn fmow_s2 fmow_8b 400 ${train_perc}"
    "base/dofa knn fmow_s2 fmow_8b 1000 ${train_perc}"
    "base/senpamae knn fmow_s2 fmow_8b 400 ${train_perc}"

    #D: s2:8b -> 18 to 20
    "base/panopticon_v3 knn fmow_8b fmow_s2 400 ${train_perc}"
    "base/dofa knn fmow_8b fmow_s2 1000 ${train_perc}"
    "base/senpamae knn fmow_8b fmow_s2 400 ${train_perc}"

    #E: 4b:s2 -> 21 to 23
    "base/panopticon_v3 knn fmow_4b fmow_s2 400 ${train_perc}"
    "base/dofa knn fmow_4b fmow_s2 1000 ${train_perc}"
    "base/senpamae knn fmow_4b fmow_s2 400 ${train_perc}"

    #F: s2:4b -> 24 to 26   
    "base/panopticon_v3 knn fmow_s2 fmow_4b 400 ${train_perc}"
    "base/dofa knn fmow_s2 fmow_4b 1000 ${train_perc}"
    "base/senpamae knn fmow_s2 fmow_4b 400 ${train_perc}"
    

    #BENv2: 10%

    # "base/dofa linear_probe benv2_s2_1b 3000 ${benv2_train_percent}"
    # "base/senpamae linear_probe benv2_s2_1b 3000 ${benv2_train_percent}"
    # "base/panopticon linear_probe benv2_s2_1b 800 ${benv2_train_percent}"

    # "base/dofa linear_probe benv2_s2_4b ${bsz_benv2} ${benv2_train_percent}" #died - run again
    # "base/senpamae linear_probe benv2_s2_4b ${bsz_benv2} ${benv2_train_percent}"
    # "base/panopticon linear_probe benv2_s2_4b ?? ${benv2_train_percent}"

    # "base/dofa linear_probe benv2_s2_10b ${bsz_benv2} ${benv2_train_percent}"
    # "base/senpamae linear_probe benv2_s2_10b ${bsz_benv2} ${benv2_train_percent}"
    # "base/panopticon linear_probe benv2_s2_10b ?? ${benv2_train_percent}"

    # "base/dofa linear_probe benv2_s2_12b ${bsz_benv2} ${benv2_train_percent}"
    # "base/senpamae linear_probe benv2_s2_12b ${bsz_benv2} ${benv2_train_percent}"
    # "base/panopticon linear_probe benv2_s2_12b ?? ${benv2_train_percent}"
    
)

########## linear probe defaults

optim=sgd
# nb_knn="[1,3,5,10,15,20,25,30]"
nb_knn="[20]"
temperature=0.07

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="0.1 0.01 0.001 0.0005"
warmup_epochs=0

########## defaults both

epochs=50
num_workers=3
check_val_every_n_epoch=5


export CUDA_VISIBLE_DEVICES=$cuda_device

# Loop through each task ID provided
for task_id in "${task_ids[@]}"; do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3
    ds_test=$4
    batch_size=$5
    train_subset=$6

    train_model_name=${model#base/}

    cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$exp_base_name/${training_mode}/${train_model_name}-null/${ds}-${ds_test} \
        +model.training_mode=$training_mode \
        ++batch_size=$batch_size \
        num_workers=$num_workers \
        num_gpus=1 \
        seed=21 \
        overwrite=$overwrite \
        +dataset_test=$ds_test \
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
                ++lr=$lr \
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
#         ++num_workers=16 \

