#!/bin/bash
#SBATCH --job-name=rlo_ablation
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=288000mb
#SBATCH --time=24:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Williamsw2179@gmail.com

set -e
echo "=== RLO Ablation Study ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

module load conda cuda/12.4
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_icml"
mkdir -p $BASE/logs $BASE/ablation

# Copy data to NVMe
echo "Copying CIFAR-100 to NVMe..."
mkdir -p $SLURM_TMPDIR/cifar
cp -r /blue/wdixon/wang.yixuan/data/cifar-100-python $SLURM_TMPDIR/cifar/ 2>/dev/null || \
    echo "CIFAR will be downloaded"

# Copy scripts
cp $BASE/*.py $SLURM_TMPDIR/
cd $SLURM_TMPDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at $(date)"
torchrun --nproc_per_node=8 --master_port=29500 train_ablation.py \
    --data $SLURM_TMPDIR/cifar \
    --output $BASE/ablation

echo "=== Completed at $(date) ==="
