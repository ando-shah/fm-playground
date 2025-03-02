#!/bin/bash
export $(cat /home/ando/fm-playground/.env)
export PYTHONPATH='.'
cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py"

fastdevrun=no
exp_base_name=T2_reg_10p_sgd_no_tnorm


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
tc_percent=0.1 #5700 samples
dt_percent=0.1 #10k samples
benv2_train_percent=0.05 #same size as corine ~ 8000 samples
eurosat_train_percent=1.0
hyperview_train_percent=1.0 #1300 samples

all_tasks=(

    #Corine-Tropical Cyclone 0-5
    "base/croma_s2_pe partial_finetune tropical_cyclone 300 ${tc_percent}"
    "base/softcon_13b_pe partial_finetune tropical_cyclone 300 ${tc_percent}"
    "base/anysat_naip_pe partial_finetune tropical_cyclone 200 ${tc_percent}"
    # "base/galileo_pe partial_finetune tropical_cyclone 100 ${tc_percent}" TBD
    "base/dofa linear_probe tropical_cyclone 3000 ${tc_percent}"
    # "base/senpamae linear_probe tropical_cyclone 3000 ${tc_percent}"  # No SRFs
    "base/panopticon_v2 linear_probe tropical_cyclone 2000 ${tc_percent}"
    "base/panopticon_v3 linear_probe tropical_cyclone 2000 ${tc_percent}"

    #Digital Typhoon 6-11
    "base/croma_s2_pe partial_finetune digital_typhoon_1b 300 ${dt_percent}"
    "base/softcon_13b_pe partial_finetune digital_typhoon_1b 300 ${dt_percent}"
    "base/anysat_naip_pe partial_finetune digital_typhoon_1b 200 ${dt_percent}"
    "base/dofa linear_probe digital_typhoon_1b 3000 ${dt_percent}"
    # "base/senpamae linear_probe digital_typhoon_1b 3000 ${tc_percent}" # No SRFs
    "base/panopticon_v2 linear_probe digital_typhoon_1b 200 ${dt_percent}"
    "base/panopticon_v3 linear_probe digital_typhoon_1b 200 ${hyperview_train_percent}"

    #Hyperview-SD 12-18
    "base/croma_s2_pe partial_finetune hyperview_sd 200 ${hyperview_train_percent}"
    "base/softcon_13b_pe partial_finetune hyperview_sd 200 ${hyperview_train_percent}"
    "base/anysat_naip_pe partial_finetune hyperview_sd 100 ${hyperview_train_percent}"
    "base/dofa linear_probe hyperview_sd 300 ${hyperview_train_percent}"
    "base/senpamae linear_probe hyperview_sd 150 ${hyperview_train_percent}"
    "base/panopticon_v2 linear_probe hyperview_sd 200 ${hyperview_train_percent}"
    "base/panopticon_v3 linear_probe hyperview_sd 200 ${hyperview_train_percent}"


    #Hyperview with no norm for target 19-25
    "base/croma_s2_pe partial_finetune hyperview_sd_no_tnorm 200 ${hyperview_train_percent}"
    "base/softcon_13b_pe partial_finetune hyperview_sd_no_tnorm 200 ${hyperview_train_percent}"
    "base/anysat_naip_pe partial_finetune hyperview_sd_no_tnorm 100 ${hyperview_train_percent}"
    "base/dofa linear_probe hyperview_sd_no_tnorm 300 ${hyperview_train_percent}"
    "base/senpamae linear_probe hyperview_sd_no_tnorm 150 ${hyperview_train_percent}"
    "base/panopticon_v2 linear_probe hyperview_sd_no_tnorm 200 ${hyperview_train_percent}"
    "base/panopticon_v3 linear_probe hyperview_sd_no_tnorm 200 ${hyperview_train_percent}"
    #Try pan and dofa with lightning    19- ERRORS
    # "base/panopticon_v3 partial_finetune digital_typhoon_1b 180 ${dt_percent}"
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

# optim=adam
optim=sgd

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="0.01 0.001 0.0005 0.0001"
warmup_epochs=5

########## defaults both

epochs=50
num_workers=2
check_val_every_n_epoch=1
val_subset=-1

export CUDA_VISIBLE_DEVICES=$cuda_device

#overwrite=$overwrite \
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


