#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=t2_chnsim_spacenet
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=0:04:00
#SBATCH --array=0

##### export env variables
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env) # horeka
#####


fastdevrun=fast
exp_base_name=t2_spacenet1_3b

all_tasks=(
    'base/dinov2 frozen_backbone spacenet1_3b 200'

    'base/dinov2_pe frozen_backbone spacenet1_8b 200'
    'base/panopticon_v3 frozen_backbone spacenet1_8b 180'
    'base/senpamae frozen_backbone spacenet1_8b 200'
    'base/croma_s2_pe frozen_backbone spacenet1_8b 200'
    'base/softcon_13b_pe frozen_backbone spacenet1_8b 200'
    'base/anysat_naip_pe frozen_backbone spacenet1_8b 200'
    'base/dofa frozen_backbone spacenet1_8b 200'
    'base/panopticon_chnsim frozen_backbone spacenet1_8b 180'

)

lrs='0.1 0.01 0.001 0.0001 0.00001 0.000001'


# Generate cross product of all_tasks and lrs
cross_product=()
for task in "${all_tasks[@]}"; do
    for lr in $lrs; do
        cross_product+=("$task $lr")
    done
done
all_tasks=("${cross_product[@]}")


########## defaults 

epochs=50
num_workers=10
check_val_every_n_epoch=1
val_subset=-1
warmup_epochs=0

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
    lrs=$5

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
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=32"
        lrs="0.1"

    elif [ $fastdevrun == 'bsz' ]; then
        echo "fastdevrun 'bsz'!"
        cmd="$cmd epochs=1 batch_size=$batch_size trainer.check_val_every_n_epoch=1"
        s=$((batch_size * 1))
        echo $s
        cmd="$cmd dataset.subset.train=$s dataset.subset.val=$s dataset.subset.test=$s"
        lrs="0.1"

    else
        echo "fastdevrun not recognized"
        exit 1
    fi


    for lr in $lrs; do
        echo "subtask with lr=$lr"
        lr_cmd="$cmd \
            +lr=$lr \
            warmup_epochs=$warmup_epochs \
            "
        echo $lr_cmd
        $lr_cmd
    done

            

    # collect results
    $PY_EXECUTABLE $REPO_PATH/geofm_src/collect_results.py $ODIR/$exp_base_name/

done