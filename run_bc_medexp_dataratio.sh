#!/bin/bash
# 测试 medium-expert 数据集在不同数据量下的表现
# 三种环境: walker2d, hopper, halfcheetah
# 数据量: 1/2 (0.5) 和 1/4 (0.25)
# 每个实验运行3次（不同seed）
# 总计: 3环境 × 2数据量 × 3种子 = 18 次实验（如不需要某些可注释）

# 设置错误时立即退出
set -e
set -o pipefail

cd /home/ssd/zml/BC_baseline

# 环境列表
ENVS=("hopper" "halfcheetah")

# 数据集：只测试 medium-expert
DATASET="medium-expert"
DATASET_SHORT="medexp"

# 数据比例
DATA_RATIOS=("0.5" "0.25")
DATA_RATIO_NAMES=("half" "quarter")

# 运行次数
NUM_RUNS=2

# 训练轮数
N_EPOCHS=1000

# Batch Size
BATCH_SIZE=2048

# 最大并行任务数
MAX_PARALLEL=2

# 存储所有要运行的任务
declare -a tasks

# 收集所有任务
for env in "${ENVS[@]}"; do
    task_name="${env}-${DATASET}-v0"
    
    for j in "${!DATA_RATIOS[@]}"; do
        ratio="${DATA_RATIOS[$j]}"
        ratio_name="${DATA_RATIO_NAMES[$j]}"
        
        for run in $(seq 1 ${NUM_RUNS}); do
            exp_name="bc-${env}-${DATASET_SHORT}-${ratio_name}-run${run}"
            seed=$((41 + run))
            tasks+=("${task_name}|${exp_name}|${seed}|${ratio}")
        done
    done
done

echo "Total experiments to run: ${#tasks[@]}"
echo "Max parallel tasks: ${MAX_PARALLEL}"
echo "Environments: ${ENVS[*]}"
echo "Dataset: ${DATASET}"
echo "Data ratios: ${DATA_RATIOS[*]}"
echo "Seeds per config: ${NUM_RUNS}"
echo ""

# 并行执行任务
running=0
task_idx=0
pids=()

while [ $task_idx -lt ${#tasks[@]} ] || [ $running -gt 0 ]; do
    # 启动新任务（如果还有任务且未达到最大并行数）
    while [ $running -lt $MAX_PARALLEL ] && [ $task_idx -lt ${#tasks[@]} ]; do
        IFS='|' read -r task exp_name seed ratio <<< "${tasks[$task_idx]}"
        
        echo "=========================================="
        echo "Starting: ${exp_name}"
        echo "Task: ${task}"
        echo "Seed: ${seed}"
        echo "Data ratio: ${ratio}"
        echo "Batch size: ${BATCH_SIZE}"
        echo "=========================================="
        
        # 后台运行训练任务
        (
            python train_bc_d4rl.py \
                --env_name=${task} \
                --exp_name=${exp_name} \
                --seed=${seed} \
                --n_epochs=${N_EPOCHS} \
                --batch_size=${BATCH_SIZE} \
                --data_ratio=${ratio}
            
            exit_code=$?
            if [ $exit_code -ne 0 ]; then
                echo "ERROR: Experiment ${exp_name} failed with exit code $exit_code!"
                exit $exit_code
            fi
        ) &
        pids+=($!)
        running=$((running + 1))
        task_idx=$((task_idx + 1))
    done

    # 等待任意一个任务完成
    if [ $running -gt 0 ]; then
        wait -n
        
        for pid in "${!pids[@]}"; do
            if ! kill -0 "${pids[$pid]}" 2>/dev/null; then
                wait "${pids[$pid]}"
                exit_code=$?
                if [ $exit_code -ne 0 ]; then
                    echo "ERROR: A background experiment failed. Exiting script."
                    exit 1
                fi
                unset 'pids[pid]'
                running=$((running - 1))
            fi
        done
        pids=("${pids[@]}")
    fi
done

echo ""
echo "All medium-expert data ratio experiments completed!"
echo "Total experiments: ${#tasks[@]}"
echo ""
echo "Experiment naming convention:"
echo "  bc-{env}-medexp-half-run{1,2,3}   -> 50% data"
echo "  bc-{env}-medexp-quarter-run{1,2,3} -> 25% data"

