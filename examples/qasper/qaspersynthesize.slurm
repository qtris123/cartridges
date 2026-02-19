#!/bin/bash
#SBATCH -A gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2 
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G 
#SBATCH --time=4:00:00
#SBATCH --job-name=qasper_synthesize
#SBATCH --output=qasper_synthesize.out
#SBATCH --error=qasper_synthesize.err


echo "=========================================="
echo "Qasper Synthesis with Tokasaurus Server"
echo "=========================================="
echo "JobID=$SLURM_JOB_ID"
echo "Partition=$SLURM_JOB_PARTITION"
echo "NodeList=$SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo ""


# GPU information
echo "Checking GPU settings..."
nvidia-smi
scontrol show job $SLURM_JOB_ID | egrep "Partition=|Account=|QOS=|TRES="
echo ""


# Activate conda environment
echo "Activating cartridges conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cartridges
echo "Python: $(which python3)"
echo ""


# Load compatible GCC for CUDA (GCC 14 causes compilation issues)
echo "Loading GCC 11.4..."
module load gcc/11.4.1
echo "GCC version: $(gcc --version | head -1)"
echo ""


# Load CUDA module for nvcc compiler (needed for flashinfer)
echo "Loading CUDA module..."
module load cuda/12.1.0
echo "CUDA version: $(nvcc --version | grep release)"
echo ""


# Set environment variables
export CARTRIDGES_DIR=/home/vo43/cartridges
export CARTRIDGES_OUTPUT_DIR=/home/vo43/cartridges/outputs
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


# GPU monitoring (separate file)
GPU_LOG="${CARTRIDGES_DIR}/examples/qasper/gpu_usage.log"  
echo "=== GPU monitor started: $(date) on $(hostname) ===" > "$GPU_LOG"
(
  while true; do
    echo "index,utilization,memory,memory.used,memory.total,temperature,power"
    date +"[%F %T]"
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw \
      --format=csv,noheader,nounits
    echo "-----"
    sleep 5
  done
) >> "$GPU_LOG" 2>&1 &
GPU_MON_PID=$!


# Trap to ensure GPU monitor is killed on exit
cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $GPU_MON_PID 2>/dev/null
    wait $GPU_MON_PID 2>/dev/null
    echo "=== GPU monitor stopped: $(date) ===" >> "$GPU_LOG"
    echo "Job finished at: $(date)"
}
trap cleanup EXIT


# Run synthesis with server management
# The Python script handles:
# - Starting tokasaurus server
# - Waiting for it to be ready
# - Running synthesis
# - Shutting down server
echo "Starting synthesis with integrated server management..."
echo ""


cd "$CARTRIDGES_DIR"
python3 examples/qasper/qasper_synthesize_with_server.py


SYNTHESIS_RC=$?


echo ""
echo "=========================================="
echo "Synthesis completed with exit code: $SYNTHESIS_RC"
echo "=========================================="


exit $SYNTHESIS_RC



