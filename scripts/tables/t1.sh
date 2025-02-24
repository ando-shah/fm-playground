##### export env variables
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env) # horeka
#####


fastdevrun=true
exp_base_name=debug2/t1
task_id=$1

all_tasks=(
    # benv2-s1
    # 'base/dinov2_pe partial_finetune benv2_s1'
    # 'base/softcon_2b linear_probe benv2_s1'
    # 'base/croma_s1 linear_probe benv2_s1'
    # 'base/dofa linear_probe benv2_s1'
    # 'base/senpamae_pe partial_finetune benv2_s1'

    # resisc45
    'base/dinov2 linear_probe resisc45'
    'base/softcon_13b_pe partial_finetune resisc45'
    'base/croma_s2_pe partial_finetune resisc45'
    'base/dofa linear_probe resisc45'
    'base/panopticon linear_probe resisc45'
    # 'base/senpamae_pe partial_finetune resisc45'
)


########## linear probe defaults

n_last_blocks_list='[1,4]'
pooling='[avgpool,cls,default]'
lrs_linear_probe='[1e-5,5e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,0.1,0.2,0.3,0.5,1,3,5,10,20]'

########## pe linear probe (=partial finetune) defaults

lrs_partial_ft="10 0.1 0.01 0.001"
warmup_epochs=0

########## defaults both

epochs=50
batch_size=500
num_workers=8
check_val_every_n_epoch=10


########## extract task specific parameters

task=${all_tasks[$task_id]}
echo "Running Task: $task"

set -- $task
model=$1
training_mode=$2
ds=$3

########## execution

cmd="$PY_EXECUTABLE $REPO_PATH/geofm_src/main.py \
    model=$model \
    dataset=$ds \
    output_dir=$ODIR/$exp_base_name/$ds/$model/$training_mode/ \
    +model.training_mode=$training_mode \
    \
    epochs=$epochs \
    \
    batch_size=$batch_size \
    num_workers=$num_workers \
    num_gpus=1 \
    seed=21 \
    "

if $fastdevrun; then
    echo "fastdevrun!"
    cmd="$cmd epochs=1 batch_size=32 trainer.check_val_every_n_epoch=1"
else
    cmd="$cmd epochs=$epochs batch_size=$batch_size trainer.check_val_every_n_epoch=$check_val_every_n_epoch"
fi


if [ $training_mode == 'linear_probe' ]; then
    
    if $fastdevrun; then
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"
    fi
    
    cmd="$cmd \
        +n_last_blocks_list=$n_last_blocks_list \
        +pooling=$pooling \
        +lr=$lrs_linear_probe"
    echo $cmd
    $cmd
        
elif [ $training_mode == 'partial_finetune' ]; then

    if $fastdevrun; then
        cmd="$cmd dataset.subset.train=64 dataset.subset.val=64 dataset.subset.test=64"
        lrs_partial_ft="0.1 0.01"
    fi

    for lr in $lrs_partial_ft; do
        echo "partial finetune with lr=$lr"
        lr_cmd="$cmd \
            +lr=$lr \
            +base_lr=-1 \
            +model.params_to_train=[] \
            warmup_epochs=$warmup_epochs \
            "
        echo $lr_cmd
        $lr_cmd
    done
fi
        
