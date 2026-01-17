#!/bin/bash
#SBATCH --job-name=rlo_abl_lr_batch
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=williamsw2179@gmail.com

set -e
hostname; date; pwd

module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_experiments"
mkdir -p $BASE/{logs,ablation}

# Copy scripts
cp $SLURM_SUBMIT_DIR/optimizers.py $BASE/
cp $SLURM_SUBMIT_DIR/train_ablation.py $BASE/

cd $BASE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run ONLY LR and Batch Size studies (the remaining ones)
# Study 1: Learning Rate Sensitivity
torchrun --nproc_per_node=8 --master_port=29540 train_ablation.py \
    --study lr \
    --data $SLURM_TMPDIR/cifar \
    --output $BASE/ablation

# Study 2: Batch Size Sensitivity
torchrun --nproc_per_node=8 --master_port=29541 train_ablation.py \
    --study batch \
    --data $SLURM_TMPDIR/cifar \
    --output $BASE/ablation

echo "Done: $(date)"
