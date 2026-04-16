#!/bin/bash
# 断点续跑：只跑剩余未完成的实验
# 已完成: walker2d 全部6个, hopper-medium 2个, hopper-medexp 2个
# 未完成: hopper-medrep 2个, halfcheetah 全部6个 = 共8个实验

# 注意：并行执行时不能使用 set -e，因为后台进程的退出码需要单独处理
set -o pipefail

cd /home/ssd/zml/BC_baseline

# mujoco_py 编译和运行所需的环境变量
export CPATH=/home/ssd/miniconda3/envs/py37/include:$CPATH
export LIBRARY_PATH=/home/ssd/miniconda3/envs/py37/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/home/ssd/miniconda3/envs/py37/lib:/home/ssd/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH

# 训练轮数
N_EPOCHS=1000

# 批次大小
BATCH_SIZE=2048

# 并行任务数
MAX_PARALLEL=2

# 直接列出剩余未完成的任务: "env_name|exp_name|seed"
declare -a tasks=(
    "hopper-medium-replay-v0|bc-hopper-medrep-seed333|333"
    "hopper-medium-replay-v0|bc-hopper-medrep-seed666|666"
    "halfcheetah-medium-v0|bc-halfcheetah-medium-seed333|333"
    "halfcheetah-medium-v0|bc-halfcheetah-medium-seed666|666"
    "halfcheetah-medium-expert-v0|bc-halfcheetah-medexp-seed333|333"
    "halfcheetah-medium-expert-v0|bc-halfcheetah-medexp-seed666|666"
    "halfcheetah-medium-replay-v0|bc-halfcheetah-medrep-seed333|333"
    "halfcheetah-medium-replay-v0|bc-halfcheetah-medrep-seed666|666"
)

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
echo "All remaining BC training experiments completed!"
echo "Total experiments: ${#tasks[@]}"
