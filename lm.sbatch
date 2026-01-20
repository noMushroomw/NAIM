#!/bin/bash
#SBATCH --job-name=rlo_lm
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=288000mb
#SBATCH --time=72:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_icml/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Williamsw2179@gmail.com

echo "=== RLO Language Model Training ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"

module load conda cuda/12.4
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_icml"
mkdir -p $BASE/logs $BASE/lm

# Verify Python scripts exist
if [ ! -f "$BASE/train_lm.py" ]; then
    echo "ERROR: train_lm.py not found in $BASE"
    exit 1
fi

# Copy scripts to work directory
cd $SLURM_TMPDIR
cp $BASE/optimizers.py .
cp $BASE/train_lm.py .

# WikiText-103 MUST be pre-downloaded (compute nodes have no internet!)
# Check multiple possible locations
mkdir -p $SLURM_TMPDIR/wikitext

FOUND=0
for WIKITEXT_SRC in \
    "/blue/wdixon/wang.yixuan/data/wikitext-103-raw" \
    "/blue/wdixon/wang.yixuan/lypcdf/wikitext-103-raw" \
    "/blue/wdixon/wang.yixuan/lypcdf/wikitext-103" \
    "/blue/wdixon/wang.yixuan/rlo_icml/wikitext-103-raw"
do
    if [ -d "$WIKITEXT_SRC" ] && [ -f "$WIKITEXT_SRC/wiki.train.raw" ]; then
        echo "Found WikiText-103 at $WIKITEXT_SRC"
        cp -r $WIKITEXT_SRC/* $SLURM_TMPDIR/wikitext/
        FOUND=1
        break
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "=============================================="
    echo "ERROR: WikiText-103 not found!"
    echo "=============================================="
    echo ""
    echo "Compute nodes do NOT have internet access."
    echo "Please download WikiText-103 manually on a login node:"
    echo ""
    echo "  cd /blue/wdixon/wang.yixuan/rlo_icml"
    echo "  wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
    echo "  unzip wikitext-103-raw-v1.zip"
    echo ""
    echo "Then resubmit this job."
    exit 1
fi

echo "WikiText files:"
ls -la $SLURM_TMPDIR/wikitext/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at $(date)"
torchrun --nproc_per_node=8 --master_port=29501 train_lm.py \
    --data $SLURM_TMPDIR/wikitext \
    --output $BASE/lm \
    --steps 100000

echo "=== Completed at $(date) ==="
