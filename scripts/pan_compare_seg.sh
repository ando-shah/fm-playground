#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a-%x.out

#SBATCH --job-name=pan_compare_seg
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=2:30:00
#SBATCH --array=0-9

##### export env variables
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env) # horeka
#####


fastdevrun=no
exp_base_name=panop_ibot_seg

# all_tasks=(

#     # m-pv4seg (rgb) 0-4
#     'base/panopticon_v3 frozen_backbone geobench_pv4ger_seg 100'

#     # m-cashew (s2) 5-12 (<2h)
#     'base/panopticon_v3 frozen_backbone geobench_cashew_12b 100'

#     # chesapeak (rgb / naip) 13-17
#     'base/panopticon_v3 frozen_backbone geobench_chesapeake_4b 100'

#     # m-nzcattle (rgb) 18-22
#     'base/panopticon_v3 frozen_backbone geobench_nzcattle 100'

#     # neontree (only rgb) 23-27
#     'base/panopticon_v3 frozen_backbone geobench_neontree 100'

#     # sa crop (s2) 28-35
#     'base/panopticon_v3 frozen_backbone geobench_sacrop_12b 100'
# )

models=(
    # 'pan/stg2/ibot_loss_high'
    # 'pan/stg2/ibot_loss_low'
    'pan/ibot_abl/lw05'
    'pan/ibot_abl/lw10'
    'pan/ibot_abl/lw20'
)

tasks=(
    'geobench_pv4ger_seg'
    'geobench_cashew_12b'
    # 'geobench_chesapeake_4b'
    # 'geobench_nzcattle'
    # 'geobench_neontree'
    'geobench_sacrop_12b'
)

all_tasks=()
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        all_tasks+=("$model $task")
    done
done


########## defaults 

epochs=50
num_workers=10
check_val_every_n_epoch=1
val_subset=-1
lrs='0.01 0.001 0.0001'
# lrs='0.1'
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
    ds=$2
    training_mode=frozen_backbone
    batch_size=100

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
        lrs="0.1 0.01"

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