
export $(cat .env)
export PYTHONPATH='.'
cmd='python geofm_src/main.py'

all_tasks=(
    ######## classification
    # 'croma_s2 linear_probe geobench_eurosat_12b'
    # 'dofa_b linear_probe geobench_eurosat'
    'dinov2 linear_probe geobench_eurosat_rgb'    
    # 'softcon_B13_b linear_probe geobench_eurosat'
    'stealth linear_probe geobench_eurosat'
    # 'anysat linear_probe geobench_eurosat_rgb'
    # 'senpamae linear_probe geobench_eurosat'
)

for task in "${all_tasks[@]}"
do
    echo "Running Task: $task"

    for lr in 0.002 0.005
    do

        set -- $task
        model=$1
        training_mode=$2
        ds=$3

        $cmd \
            model=$model \
            dataset=$ds \
            +model.training_mode=$training_mode \
            \
            lr=0.002 \
            epochs=1 \
            warmup_epochs=0 \
            \
            batch_size=64 \
            num_workers=8 \
            num_gpus=1 \
            seed=13 \
            +trainer.fast_dev_run=True \
            \
            dataset.subset.train=0.4 # just as example how to use subset
            
    done
done