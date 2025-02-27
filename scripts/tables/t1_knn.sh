#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
##SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a.out

#SBATCH --job-name=eval
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=0:10:00
#SBATCH --array=9



##### export env variables
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env) # horeka
#####


fastdevrun=no
exp_base_name=r02_knn

all_tasks=(

    # resisc45 (50e = 1h30 (lp))
    'base/dofa knn resisc45 900'
    'base/dinov2 knn resisc45 900'
    'base/panopticon_v2 knn resisc45 400'

    # eurosat (50e = 0h15 (lp))
    'base/croma_s2 knn geobench_eurosat_12b 900'
    'base/dofa knn geobench_eurosat_13b 900'
    'base/dinov2 knn geobench_eurosat_rgb 900'
    'base/softcon_13b knn geobench_eurosat_13b 900'
    'base/panopticon_v2 knn geobench_eurosat_13b 200'
    'base/galileo_s2 knn geobench_eurosat_10b 200 '
    'base/anysat_s2 knn geobench_eurosat_10b 50 '

    # # benv2-s2 (50e, 10% subset = 0h20)
    # 'base/croma_s2 linear_probe benv2_s2_12b 900'
    # 'base/dofa linear_probe benv2_s2_12b 900'
    # 'base/dinov2 linear_probe benv2_s2_rgb 900'
    # 'base/panopticon_v2 linear_probe benv2_s2_12b 200'

    # # benv2-s1 
    # 'base/croma_s1 linear_probe benv2_s1 900'
    # 'base/dofa linear_probe benv2_s1 900'
    # 'base/softcon_2b linear_probe benv2_s1 900'
    # 'base/panopticon_v2 linear_probe benv2_s1 400'

)


########## defaults

temperature=0.07
nb_knn="[20]"
num_workers=8

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
        cmd="$cmd epochs=$epochs batch_size=$batch_size"

    elif [ $fastdevrun == 'fast' ]; then
        echo "fastdevrun 'fast'!"
        cmd="$cmd batch_size=32"
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"

    elif [ $fastdevrun == 'bsz' ]; then
        echo "fastdevrun 'bsz'!"
        cmd="$cmd batch_size=$batch_size"
        s=$((batch_size * 1))
        echo $s
        cmd="$cmd dataset.subset.train=$s dataset.subset.val=$s dataset.subset.test=$s"

    else
        echo "fastdevrun not recognized"
        exit 1
    fi

        
    cmd="$cmd \
        +temperature=$temperature \
        +nb_knn=$nb_knn \
        "
    echo $cmd
    $cmd

    # collect results
    $PY_EXECUTABLE $REPO_PATH/geofm_src/collect_results.py $ODIR/$exp_base_name/

done