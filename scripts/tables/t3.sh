#!/bin/bash
export $(cat /home/ando/fm-playground/.env)
export PYTHONPATH='.'
cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py"

fastdevrun=no
exp_base_name=t2_brickkiln
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

    #DOFA: Corine 0-2
    "base/dofa linear_probe corine_1b 3000 ${corine_train_percent}"
    "base/senpamae linear_probe corine_1b 2048 ${corine_train_percent}"
    "base/panopticon linear_probe corine_1b 800 ${corine_train_percent}"

    #Corine 3-5
    "base/dofa linear_probe corine_4b 2048 ${corine_train_percent}"
    "base/senpamae linear_probe corine_4b 1200 ${corine_train_percent}"
    "base/panopticon linear_probe corine_4b 500 ${corine_train_percent}"

    #Corine 6-8
    "base/dofa linear_probe corine_10b 2048 ${corine_train_percent}"
    "base/senpamae linear_probe corine_10b 500 ${corine_train_percent}"
    "base/panopticon linear_probe corine_10b 280 ${corine_train_percent}"

    #Corine 9-11
    "base/dofa linear_probe corine_21b 1500 ${corine_train_percent}"
    "base/senpamae linear_probe corine_21b 300 ${corine_train_percent}"
    "base/panopticon linear_probe corine_21b 130 ${corine_train_percent}"

    #Corine 12-14   
    "base/dofa linear_probe corine_50b 600 ${corine_train_percent}" #anything more than 512 is throttled by CPU RAM
    "base/senpamae linear_probe corine_50b 60 ${corine_train_percent}"
    "base/panopticon linear_probe corine_50b 60 1.0"

    # low prio: 15-17
    "base/dofa linear_probe corine_202b 100 ${corine_train_percent}"
    "base/senpamae linear_probe corine_202b 15 ${corine_train_percent}"
    "base/panopticon linear_probe corine_202b 15 ${corine_train_percent}"

    # Corine kNN :(
    # "base/dofa knn corine_1b 10000 ${corine_train_percent}"
    # "base/senpamae knn corine_1b 8000 ${corine_train_percent}"
    # "base/panopticon knn corine_1b 2400 ${corine_train_percent}"

    # "base/dofa knn corine_4b 5000 ${corine_train_percent}"
    # "base/senpamae knn corine_4b 3000 ${corine_train_percent}"
    # "base/panopticon knn corine_4b 1000 ${corine_train_percent}"

    # "base/dofa knn corine_10b 3000 ${corine_train_percent}"
    # "base/senpamae knn corine_10b 1000 ${corine_train_percent}"
    # "base/panopticon knn corine_10b 500 ${corine_train_percent}"
    
    # "base/dofa knn corine_21b 1024 ${corine_train_percent}"
    # "base/senpamae knn corine_21b 1024 ${corine_train_percent}"
    # "base/panopticon knn corine_21b 150 ${corine_train_percent}"

    
    # Eurosat kNN 18-21
    "base/dofa knn geobench_eurosat_1b 30000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_eurosat_1b 30000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_eurosat_1b 20000 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_eurosat_1b 20000 ${eurosat_train_percent}"

    # Eurosat kNN 22-25
    "base/dofa knn geobench_eurosat_4b 20000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_eurosat_4b 20000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_eurosat_4b 1000 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_eurosat_4b 1000 ${eurosat_train_percent}"
    # Eurosat kNN 26-29
    "base/dofa knn geobench_eurosat_10b 15000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_eurosat_10b 1000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_eurosat_10b 400 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_eurosat_10b 400 ${eurosat_train_percent}"

    # Eurosat kNN 30-33 
    "base/dofa knn geobench_eurosat_12b 15000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_eurosat_12b 1000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_eurosat_12b 300 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_eurosat_12b 300 ${eurosat_train_percent}"



    # Brick Kiln kNN 34-37
    "base/dofa knn geobench_brick_kiln_1b 30000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_brick_kiln_1b 30000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_brick_kiln_1b 20000 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_brick_kiln_1b 20000 ${eurosat_train_percent}"

    # Brick Kiln kNN 38-41  
    "base/dofa knn geobench_brick_kiln_4b 20000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_brick_kiln_4b 20000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_brick_kiln_4b 1000 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_brick_kiln_4b 1000 ${eurosat_train_percent}"

    # Brick Kiln kNN 42-45
    "base/dofa knn geobench_brick_kiln_10b 15000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_brick_kiln_10b 1000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_brick_kiln_10b 400 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_brick_kiln_10b 400 ${eurosat_train_percent}"
    # Brick Kiln kNN 46-49
    "base/dofa knn geobench_brick_kiln_12b 15000 ${eurosat_train_percent}"
    "base/senpamae knn geobench_brick_kiln_12b 1000 ${eurosat_train_percent}"
    "base/panopticon knn geobench_brick_kiln_12b 300 ${eurosat_train_percent}"
    "base/panopticon_v3 knn geobench_brick_kiln_12b 300 ${eurosat_train_percent}"
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

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both

epochs=50
num_workers=4
check_val_every_n_epoch=10
val_subset=-1
nb_knn="[20]"


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


