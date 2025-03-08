#!/bin/bash
export $(cat /home/ando/fm-playground/.env)
export PYTHONPATH='.'
cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py"

fastdevrun=no
exp_base_name=cross_sens/benv2
# exp_base_name=cross_sens/benv2 
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
corine_train_percent=0.1
benv2_train_percent=0.1 #same size as corine ~ 8000 samples
eurosat_train_percent=1.0
accel_cls_blk_indices="[10,11]"
last_n_blocks="[1]"

bsz_softcon=2000
bsz_croma=2000
bsz_dofa=2000
bsz_panopticon=230
bsz_galileo=650

all_tasks=(

    # Baselines: s1 -> 0 to 4
    "base/panopticon_v3 null linear_probe benv2_s1 benv2_s1 $bsz_panopticon ${benv2_train_percent}"
    "base/dofa null linear_probe benv2_s1 benv2_s1 $bsz_dofa ${benv2_train_percent}"
    "base/croma_s1 null linear_probe benv2_s1 benv2_s1 $bsz_croma ${benv2_train_percent}"
    "base/galileo_s1_120 null linear_probe benv2_s1 benv2_s1 $bsz_galileo ${benv2_train_percent}"
    "base/softcon_2b null linear_probe benv2_s1_scnorm benv2_s1_scnorm $bsz_softcon ${benv2_train_percent}" #REDO


    # Baselines: s2 -> 5 to 9
    "base/panopticon_v3 linear_probe benv2_s2_12b benv2_s2_12b $bsz_panopticon ${benv2_train_percent}" #REDO
    "base/dofa linear_probe benv2_s2_12b benv2_s2_12b $bsz_dofa ${benv2_train_percent}" #REDO
    "base/croma_s2 linear_probe benv2_s2_12b benv2_s2_12b $bsz_croma ${benv2_train_percent}" #REDO
    "base/galileo_s2_120 null linear_probe benv2_s2_10b benv2_s2_10b $bsz_galileo ${benv2_train_percent}" #REDO
    "base/softcon_13b null linear_probe benv2_s2_13b_scnorm benv2_s2_13b_scnorm $bsz_softcon ${benv2_train_percent}" #REDO

    # A: s1:s2 -> 10 to 14
    "base/panopticon_v3 null linear_probe benv2_s1 benv2_s2_12b $bsz_panopticon ${benv2_train_percent}" #REDO
    "base/dofa null linear_probe benv2_s1 benv2_s2_12b $bsz_dofa ${benv2_train_percent}"
    "base/croma_s1 base/croma_s2 linear_probe benv2_s1 benv2_s2_12b $bsz_croma ${benv2_train_percent}"
    "base/galileo_s1_120 base/galileo_s2_120 linear_probe benv2_s1 benv2_s2_10b $bsz_galileo ${benv2_train_percent}"
    "base/softcon_2b base/softcon_13b linear_probe benv2_s1_scnorm benv2_s2_13b_scnorm $bsz_softcon ${benv2_train_percent}"
    
    #B: s2:s1 -> 15 to 19
    "base/panopticon_v3 null linear_probe benv2_s2_10b benv2_s1 $bsz_panopticon ${benv2_train_percent}"
    "base/dofa null linear_probe benv2_s2_12b benv2_s1 $bsz_dofa ${benv2_train_percent}"
    "base/croma_s2 base/croma_s1 linear_probe benv2_s2_12b benv2_s1 $bsz_croma ${benv2_train_percent}"
    "base/galileo_s2_120 base/galileo_s1_120 linear_probe benv2_s2_10b benv2_s1 $bsz_galileo ${benv2_train_percent}"
    "base/softcon_13b base/softcon_2b linear_probe benv2_s2_13b_scnorm benv2_s1_scnorm $bsz_softcon ${benv2_train_percent}"

    
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
check_val_every_n_epoch=10


export CUDA_VISIBLE_DEVICES=$cuda_device

# Loop through each task ID provided
for task_id in "${task_ids[@]}"; do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set -- $task
    model=$1
    test_model=$2
    training_mode=$3
    ds=$4
    ds_test=$5
    batch_size=$6
    train_subset=$7

    #stip base/ from model
    train_model_name=${model#base/}
    test_model_name=${test_model#base/}


    # echo $model
    # echo $test_model
    # echo $training_mode
    # echo $ds
    # echo $ds_test
    # echo $batch_size
    # echo $train_subset
    # exit 1
    #        +n_last_blocks_list=[4] \
    #++model.accel_cls_blk_indices=$accel_cls_blk_indices \

    cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$exp_base_name/${training_mode}/${train_model_name}-${test_model_name}/${ds}-${ds_test} \
        +model.training_mode=$training_mode \
        ++batch_size=$batch_size \
        num_workers=$num_workers \
        num_gpus=1 \
        seed=21 \
        overwrite=$overwrite \
        +dataset_test=$ds_test \
        +model_test=$test_model \
        +n_last_blocks_list=$last_n_blocks \
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
    echo "Test Model: $test_model"
    echo "Training Mode: $training_mode"
    echo "Dataset: $ds"
    echo "Dataset Test: $ds_test"
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
        ++num_workers=16 \
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


