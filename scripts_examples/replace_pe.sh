

export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'



all_tasks=(

    'base/dinov2_pe partial_finetune geobench_eurosat'    
    'base/softcon_B13_pe partial_finetune geobench_eurosat_12b'    
    'base/croma_s2_pe partial_finetune geobench_eurosat_12b'
)

suffix='debug/11'

for task in "${all_tasks[@]}"
do
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3

    $cmd \
        model=$model \
        dataset=$ds \
        output_dir=$ODIR/$model/$training_mode/$ds/$suffix \
        +model.training_mode=$training_mode \
        \
        +lr=1e-3 \
        +base_lr=-1\
        epochs=1 \
        warmup_epochs=0 \
        \
        batch_size=64 \
        num_workers=8 \
        num_gpus=1 \
        seed=13 \
        +trainer.fast_dev_run=False \
        \
        +model.params_to_train=[]
        # dataset.subset.train=0.4 # just as example how to use subset 
        
done