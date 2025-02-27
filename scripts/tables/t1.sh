#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
##SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a.out

#SBATCH --job-name=t1
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=4:00:00
#SBATCH --array=20,18
##SBATCH --array=18-20,24-25

##### export env variables
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env) # horeka
#####


fastdevrun=fast
exp_base_name=debug_norm2

all_tasks=(

    # m-eurosat, 0-7 (50e = 0h15 (lp))
    'base/dinov2 linear_probe geobench_eurosat_rgb 800'
    'base/croma_s2 linear_probe geobench_eurosat_12b 900'
    'base/softcon_13b linear_probe geobench_eurosat_13b 900'
    'base/anysat_s2 linear_probe geobench_eurosat_10b 900'
    'galileo'
    'base/senpamae linear_probe geobench_eurosat_13b 900'
    'base/dofa linear_probe geobench_eurosat_13b 900'
    'base/panopticon_v2 linear_probe geobench_eurosat_13b 200'

    # resisc45, 8-11 (50e = 1h30 (lp))
    'base/dinov2 linear_probe resisc45 900'
    'base/senpamae linear_probe resisc45 900'
    'base/dofa linear_probe resisc45 900'
    'base/panopticon_v2 linear_probe resisc45 400'

    # benv2-s1, 12-17
    'base/croma_s1 linear_probe benv2_s1 900'
    'base/softcon_2b linear_probe benv2_s1 900'
    'base/anysat_s1_asc linear_probe benv2_s1 900'
    'galileo'
    'base/dofa linear_probe benv2_s1 900'
    'base/panopticon_v2 linear_probe benv2_s1 400'

    # benv2-s2, 18-25
    'base/dinov2 linear_probe benv2_rgb 900'
    'base/croma_s2 linear_probe benv2_s2_12b 900'
    'base/softcon_13b linear_probe benv2_s2_13b 800'
    'base/anysat_s2 linear_probe benv2_s2_10b 900'
    'galileo'
    'base/senpamae linear_probe benv2_s2_4b 900'
    'base/dofa linear_probe benv2_s2_12b 900'
    'base/panopticon_v2 linear_probe benv2_s2_12b 200'

    # forestnet, 26-31
    'base/dinov2 linear_probe geobench_forestnet_rgb 900'
    'base/anysat landsat???'
    'galileo'
    'base/senpamae linear_probe geobench_forestnet_6b 900'
    'base/dofa linear_probe geobench_forestnet_6b 900'
    'base/panopticon_v2 linear_probe geobench_forestnet_6b 200'

    # fmow-wv, 32-34
    'base/senpamae linear_probe fmow_8b 900'
    'base/dofa linear_probe fmow_8b'
    'base/panopticon_v2 linear_probe fmow_8b'


    # # corine-sd (previous t1)
    # 'base/dofa linear_probe corine_sd 900'
    # 'base/panopticon_v2 linear_probe corine_sd 200'
    # # 'base/senpamae linear_probe corine_sd_4b 900'

    # # corine-21 (previous t1)
    # 'base/dofa linear_probe corine_21b 900'
    # 'base/panopticon_v2 linear_probe corine_21b 130'
)


########## linear probe defaults

optim=sgd

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both

epochs=50
num_workers=16
check_val_every_n_epoch=10
val_subset=-1

# epochs=10
# num_workers=12
# check_val_every_n_epoch=1
# val_subset=0.1

########## get tasks

if [ $# -eq 0 ]; then
    if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
        task_ids=($SLURM_ARRAY_TASK_ID)
    else
        task_ids=($(seq 0 $((${#all_tasks[@]}-1))))
    fi
else
    task_ids=("$@")
fi


########## execution

for task_id in "${task_ids[@]}"; do

    task=${all_tasks[$task_id]}
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3
    batch_size=$4


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
        cmd="$cmd epochs=$epochs batch_size=$batch_size trainer.check_val_every_n_epoch=$check_val_every_n_epoch dataset.subset.val=$val_subset"

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
    fi
            

    # collect results
    $PY_EXECUTABLE $REPO_PATH/geofm_src/collect_results.py $ODIR/$exp_base_name/

done