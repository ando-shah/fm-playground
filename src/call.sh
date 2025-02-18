export $(cat .env)
export PYTHONPATH='/home/hk-project-pai00028/tum_mhj8661/code/fm-playground:/home/hk-project-pai00028/tum_mhj8661/code/PanOpticOn'

all_tasks=(
    'new/anysat linear_probe geobench_eurosat'
    'new/croma linear_probe geobench_eurosat'
    'new/dinov2 linear_probe geobench_eurosat'
    # 'new/dofa_b linear_probe geobench_eurosat'
    'new/satmae linear_probe geobench_eurosat'
    'new/scalemae linear_probe geobench_eurosat'
    'new/senpamae linear_probe geobench_eurosat'
    'new/softcon linear_probe geobench_eurosat'
    'new/stealth linear_probe geobench_eurosat'
)

for task in "${all_tasks[@]}"
do
    echo "Running Task: $task"

    set -- $task
    model=$1
    training_mode=$2
    ds=$3

    python src/main.py \
        model=$model \
        dataset=$ds \
        training_mode=$training_mode \
        \
        lr=0.002 \
        epochs=1 \
        warmup_epochs=0 \
        \
        batch_size=64 \
        num_workers=8 \
        num_gpus=1 \
        seed=13 \

done