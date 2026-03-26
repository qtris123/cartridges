#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00 
#SBATCH --job-name=synthesize
#SBATCH --output=synthesize.out
#SBATCH --error=synthesize.err
 
echo "Checking GPU settings..."
nvidia-smi
echo "JobID=$SLURM_JOB_ID"
echo "Partition=$SLURM_JOB_PARTITION"
echo "NodeList=$SLURM_JOB_NODELIST"
scontrol show job $SLURM_JOB_ID | egrep "Partition=|Account=|QOS=|TRES="


# GPU monitoring (separate file)
GPU_LOG="gpu_usage.log"
(
  echo "=== GPU monitor started: $(date) on $(hostname) ==="
  while true; do
    echo "index,utilization,memory,memory.used,memory.total,temperature,power"
    date +"[%F %T]"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    echo "-----"
    sleep 1
  done
) >> "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!  

conda activate cartridges
python3 arxiv_synthesize.py
TRAIN_RC=$?

kill $GPU_MON_PID 2>/dev/null
wait $GPU_MON_PID 2>/devnull
echo "=== GPU monitor stopped: $(date) ===" >> "$GPU_LOG"

exit $TRAIN_RC

# Check GPU settings
# echo "Checking GPU settings..."
# nvidia-smi

#tksrs model=Qwen/Qwen3-4b kv_cache_num_tokens='(512 * 1024)' max_topk_logprobs=20 dp_size=1
# python3 examples/arxiv/arxiv_train.py
#