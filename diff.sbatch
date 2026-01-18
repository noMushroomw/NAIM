#!/bin/bash
#SBATCH --job-name=rlo_diffusion
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=288000mb
#SBATCH --time=120:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Williamsw2179@gmail.com

set -e
echo "=== RLO Diffusion Training ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

module load conda cuda/12.4
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_icml"
mkdir -p $BASE/logs $BASE/diffusion

# Copy ImageNet 64x64 to NVMe
echo "Copying ImageNet to NVMe..."
mkdir -p $SLURM_TMPDIR/imagenet

IMAGENET_SRC="/blue/wdixon/wang.yixuan/data/imagenet"
if [ -d "$IMAGENET_SRC" ]; then
    echo "Copying from $IMAGENET_SRC..."
    rsync -a --info=progress2 $IMAGENET_SRC/train $SLURM_TMPDIR/imagenet/ &
    rsync -a --info=progress2 $IMAGENET_SRC/val $SLURM_TMPDIR/imagenet/ &
    wait
else
    echo "ERROR: ImageNet not found at $IMAGENET_SRC"
    exit 1
fi

# Copy scripts
cp $BASE/*.py $SLURM_TMPDIR/
cd $SLURM_TMPDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at $(date)"
torchrun --nproc_per_node=8 --master_port=29503 train_diffusion.py \
    --data $SLURM_TMPDIR/imagenet \
    --output $BASE/diffusion \
    --epochs 100 \
    --resolution 64

echo "=== Completed at $(date) ==="
