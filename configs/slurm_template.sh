#!/bin/bash
#SBATCH --job-name=kg_net_sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/sft_%j.out
#SBATCH --error=logs/sft_%j.err
#
# If your group/allocation requires it, uncomment and fill these in:
##SBATCH --account=<YOUR_ACCOUNT>
##SBATCH --mail-type=begin,end,fail
##SBATCH --mail-user=<YOUR_NETID>@princeton.edu

# ===================================================================
# Della SLURM Template for Network Curriculum SFT
# 
# Assumptions:
# - Submit from the repository root with: sbatch configs/slurm_template.sh
# - The conda environment lives at /scratch/gpfs/JHA/hp9084/conda_envs/kg-si-rl
# - The converted HF dataset exists at datasets/network_curriculum
# - For Qwen3-14B, prefer full A100/H200 GPUs rather than MIG slices
# ===================================================================

set -euo pipefail

module purge
module load anaconda3/2025.6
module load cudatoolkit/12.8
module load gcc/11

export SCRATCH_BASE=/scratch/gpfs/JHA/hp9084
export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip_cache"
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_BASE}/hf_datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

conda activate "${SCRATCH_BASE}/conda_envs/kg-si-rl"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export NCCL_DEBUG=WARN
export CUDA_MODULE_LOADING=EAGER
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.8,expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

NPROC="${SLURM_GPUS_ON_NODE:-4}"

echo "Running on $(hostname)"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "GPUs: ${NPROC}"
echo "HF_HOME: ${HF_HOME}"

torchrun --standalone --nproc_per_node="${NPROC}" \
  sft_training.py \
  --model_name "Qwen/Qwen3-14B" \
  --dataset_path "datasets/network_curriculum" \
  --output_dir "sft_models/qwen3-14b-network-lora" \
  --block_size 2048 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --num_train_epochs 20 \
  --logging_steps 10 \
  --save_steps 500 \
  --bf16 True \
  --deepspeed configs/deepspeed_config.json

echo "Training completed!"

# ===================================================================
# Notes:
# 
# 1. To change GPU count, edit --gres=gpu:N. torchrun reads
#    SLURM_GPUS_ON_NODE when available.
# 2. If your dataset only exists as JSON, run:
#      python convert_curriculum_dataset.py --overwrite
# 3. For memory issues, reduce --block_size or increase gradient accumulation.
# 4. For RL training, switch to rl_training.py and add --sft_checkpoint_path.
# ===================================================================
