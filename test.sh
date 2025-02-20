
export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'

all_tasks=(
    ######## classification
    'base/croma_s2 linear_probe geobench_eurosat_12b'
    'base/dofa linear_probe geobench_eurosat'
    'base/dinov2 linear_probe geobench_eurosat_rgb'    
    'base/softcon_B13 linear_probe geobench_eurosat'
    'base/panopticon linear_probe geobench_eurosat'
    'base/senpamae linear_probe geobench_eurosat'

    ######## not checked
    # 'base/croma_s1 linear_probe geobench_eurosat_2b'
    # 'base/anysat linear_probe geobench_eurosat_rgb'
)

# flags disregarded with linear_probe: lr, warmup_epochs, trainer.*, num_gpus

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
        +model.training_mode=$training_mode \
        \
        lr=0.005 \
        epochs=1 \
        warmup_epochs=0 \
        \
        batch_size=64 \
        num_workers=8 \
        num_gpus=1 \
        seed=13 \
        +trainer.fast_dev_run=True 
        # dataset.subset.train=0.4 # just as example how to use subset 

done