#!/bin/bash
#SBATCH --job-name=rlo_vit_b16
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=288000mb
#SBATCH --time=144:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Williamsw2179@gmail.com

set -e
echo "=== RLO ViT-B/16 Classification ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

module load conda cuda/12.4
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_icml"
mkdir -p $BASE/logs $BASE/classification

# Copy ImageNet to NVMe
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

echo "ImageNet structure:"
ls -la $SLURM_TMPDIR/imagenet/

# Copy scripts
cp $BASE/*.py $SLURM_TMPDIR/
cd $SLURM_TMPDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at $(date)"
torchrun --nproc_per_node=8 --master_port=29504 train_classification.py \
    --data $SLURM_TMPDIR/imagenet \
    --output $BASE/classification \
    --epochs 90

echo "=== Completed at $(date) ==="
