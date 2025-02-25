#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
##SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%A_%a.out

#SBATCH --job-name=eval
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20        # default: 38
#SBATCH --time=1:30:00
#SBATCH --array=0-2,8-15

task_id=$SLURM_ARRAY_TASK_ID

##### export env variables
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env) # horeka
#####


fastdevrun=no
exp_base_name=r02

all_tasks=(

    # resisc45 (50e = 1h30 (lp))
    'base/dofa linear_probe resisc45 900'
    'base/dinov2 linear_probe resisc45 900'
    'base/panopticon linear_probe resisc45 400'

    # eurosat (50e = 0h15 (lp))
    'base/croma_s2 linear_probe geobench_eurosat_12b 900'
    'base/dofa linear_probe geobench_eurosat_13b 900'
    'base/dinov2 linear_probe geobench_eurosat_rgb 900'
    'base/softcon_13b linear_probe geobench_eurosat_13b 900'
    'base/panopticon linear_probe geobench_eurosat_13b 200'

    # benv2-s2 (50e, 10% subset = 0h20)
    'base/croma_s2 linear_probe benv2_s2_12b 900'
    'base/dofa linear_probe benv2_s2_12b 900'
    'base/dinov2 linear_probe benv2_s2_rgb 900'
    'base/panopticon linear_probe benv2_s2_12b 200'

    # benv2-s1 
    'base/croma_s1 linear_probe benv2_s1 900'
    'base/dofa linear_probe benv2_s1 900'
    'base/softcon_2b linear_probe benv2_s1 900'
    'base/panopticon linear_probe benv2_s1 400'

)


########## linear probe defaults

n_last_blocks_list='[1,4]'
pooling='[avgpool,cls,default]'
lrs_linear_probe='[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.5,1,3,5,10,20]'

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both

epochs=50
num_workers=8
check_val_every_n_epoch=100


########## extract task specific parameters

task=${all_tasks[$task_id]}
echo "Running Task: $task"

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
    ++batch_size=$batch_size \
    num_workers=$num_workers \
    num_gpus=1 \
    seed=21 \
    "

if [ $fastdevrun == 'no' ]; then
    cmd="$cmd epochs=$epochs batch_size=$batch_size trainer.check_val_every_n_epoch=$check_val_every_n_epoch"

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
        +n_last_blocks_list=$n_last_blocks_list \
        +pooling=$pooling \
        +lr=$lrs_linear_probe"
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