#!/bin/bash
# 使用 stable-baselines3 和 imitation 库运行 BC offline 训练
# 三种环境: walker2d, hopper, halfcheetah
# 三种数据集: medium, medium-expert (medexp), medium-replay (medrep)
# 每个实验运行3次（使用不同的seed）

# 注意：并行执行时不能使用 set -e，因为后台进程的退出码需要单独处理
set -o pipefail

cd /home/ssd/zml/BC_baseline

# 环境列表
ENVS=("walker2d" "hopper" "halfcheetah")

# 数据集列表 (medium, medium-expert, medium-replay)
DATASETS=("medium" "medium-expert" "medium-replay")
DATASET_SHORT=("medium" "medexp" "medrep")

# 运行次数
NUM_RUNS=3

# 训练轮数
N_EPOCHS=1000

# 批次大小
BATCH_SIZE=2048

# 并行任务数
MAX_PARALLEL=2

# 收集所有要运行的任务
declare -a tasks=()

for env in "${ENVS[@]}"; do
    for i in "${!DATASETS[@]}"; do
        dataset="${DATASETS[$i]}"
        dataset_short="${DATASET_SHORT[$i]}"
        task="${env}-${dataset}-v0"
        
        for run in $(seq 1 ${NUM_RUNS}); do
            exp_name="bc-${env}-${dataset_short}-run${run}"
            seed=$((41 + run))
            tasks+=("${task}|${exp_name}|${seed}")
        done
    done
done

echo "Total experiments: ${#tasks[@]}"
echo "Running ${MAX_PARALLEL} tasks in parallel"
echo ""

# 并行执行任务
running=0
task_idx=0

while [ $task_idx -lt ${#tasks[@]} ] || [ $running -gt 0 ]; do
    # 启动新任务（如果还有任务且未达到最大并行数）
    while [ $running -lt $MAX_PARALLEL ] && [ $task_idx -lt ${#tasks[@]} ]; do
        IFS='|' read -r task exp_name seed <<< "${tasks[$task_idx]}"
        
        echo "=========================================="
        echo "Starting: ${exp_name}"
        echo "Task: ${task}"
        echo "Seed: ${seed}"
        echo "Batch size: ${BATCH_SIZE}"
        echo "=========================================="
        
        # 后台运行训练任务
        (
            python train_bc_d4rl.py \
                --env_name=${task} \
                --exp_name=${exp_name} \
                --seed=${seed} \
                --n_epochs=${N_EPOCHS} \
                --batch_size=${BATCH_SIZE}
            
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "ERROR: Experiment ${exp_name} failed with exit code $exit_code!"
                exit $exit_code
            else
                echo "Completed: ${exp_name}"
            fi
        ) &
        
        running=$((running + 1))
        task_idx=$((task_idx + 1))
    done
    
    # 等待任意一个任务完成
    wait -n
    running=$((running - 1))
    
    # 检查是否有任务失败
    if [ $? -ne 0 ]; then
        echo "ERROR: A task failed! Stopping all tasks..."
        # 等待所有后台任务
        wait
        exit 1
    fi
done

# 等待所有剩余任务完成
wait

echo ""
echo "All BC training experiments completed!"
echo "Total experiments: ${#tasks[@]}"

