#!/bin/bash
#SBATCH --account=ai-forge
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=56G
#SBATCH --time=1:00:00
#SBATCH --job-name=run_benchmamrk
#SBATCH --output=run_benchmamrk.out
#SBATCH --error=run_benchmamrk.err


echo "============================================"
echo "Job ID  : $SLURM_JOB_ID"
echo "Node    : $SLURM_NODELIST"
echo "GPU     : $CUDA_VISIBLE_DEVICES"
echo "Time    : $(date)"
echo "============================================"

set -euo pipefail

nvidia-smi

# --- [FIXED] GPU MONITORING START ---
GPU_LOG="gpu_usage_${SLURM_JOB_ID}.csv"

# 1. Initialize the header
echo "timestamp, gpu_idx, gpu_util, mem_used, mem_total" > "$GPU_LOG"

# 2. Start monitoring every 3 seconds
# Added 'stdbuf -oL' to force line-buffering so you can see logs in real-time
# Added 'index' to query so you can distinguish between the 2 GPUs
stdbuf -oL nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits -l 3 >> "$GPU_LOG" &

MONITOR_PID=$!
echo "GPU Monitoring started (PID: $MONITOR_PID) every 3s, logging to $GPU_LOG"
# --- CONDA SETUP ---
# Source the conda profile script to make the 'conda' command available
source /apps/external/conda/2025.09/etc/profile.d/conda.sh

# Activate your environment
conda activate cartridges_env

export CARTRIDGES_DIR=/home/vo43/cartridges
export CARTRIDGES_OUTPUT_DIR=/home/vo43/cartridges/output

# -------------------

# Now run your python script
python /home/vo43/cartridges/examples/run_benchmark.py benchmark.dataset_name=mmlu benchmark.subset=college_physics benchmark.max_samples=50
#python run_benchmark.py dataset_name=mmlu subset=college_physics max_samples=50
# --- [NEW] CLEANUP MONITORING ---
# --- [FIXED] CLEANUP MONITORING ---
# Use a trap or explicit kill to ensure the background process dies
if ps -p $MONITOR_PID > /dev/null; then
    kill $MONITOR_PID
fi
echo "============================================"
echo "  Benchmark complete. GPU log saved to $GPU_LOG"
echo "  Time: $(date)"
echo "============================================"