#!/bin/bash
#SBATCH --job-name=rlo_abl_remaining
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=8:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.err

set -e
hostname; date; pwd
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
df -h $SLURM_TMPDIR

module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_experiments"
mkdir -p $BASE/{logs,ablation}

# Use LOCAL NVMe for CIFAR
LOCAL_DATA="$SLURM_TMPDIR/cifar"
mkdir -p $LOCAL_DATA

# Copy scripts
cp $SLURM_SUBMIT_DIR/optimizers.py $BASE/
cp $SLURM_SUBMIT_DIR/train_ablation.py $BASE/

cd $BASE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings for stability
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Run remaining ablation studies (LR study already completed)
# Study 2: Batch size sensitivity
echo "========== Running Batch Size Study =========="
torchrun --nproc_per_node=8 --master_port=29540 train_ablation.py \
    --study batch \
    --data $LOCAL_DATA \
    --output $BASE/ablation

# Study 3: Eta study (SmoothLiftedRLO only)
echo "========== Running Eta Study =========="
torchrun --nproc_per_node=8 --master_port=29541 train_ablation.py \
    --study eta \
    --data $LOCAL_DATA \
    --output $BASE/ablation

# Study 4: Lambda_b study
echo "========== Running Lambda Study =========="
torchrun --nproc_per_node=8 --master_port=29542 train_ablation.py \
    --study lambda \
    --data $LOCAL_DATA \
    --output $BASE/ablation

echo "Done: $(date)"
