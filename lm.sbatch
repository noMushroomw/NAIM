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

set -e
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
    echo "Please upload all .py files to $BASE first!"
    exit 1
fi

# Copy scripts to work directory
cd $SLURM_TMPDIR
cp $BASE/optimizers.py .
cp $BASE/train_lm.py .

# Copy or download WikiText-103
mkdir -p $SLURM_TMPDIR/wikitext

WIKITEXT_SRC="/blue/wdixon/wang.yixuan/data/wikitext-103-raw"
WIKITEXT_SRC2="/blue/wdixon/wang.yixuan/lypcdf/wikitext-103-raw"

if [ -d "$WIKITEXT_SRC" ]; then
    echo "Copying WikiText-103 from $WIKITEXT_SRC..."
    cp -r $WIKITEXT_SRC/* $SLURM_TMPDIR/wikitext/
elif [ -d "$WIKITEXT_SRC2" ]; then
    echo "Copying WikiText-103 from $WIKITEXT_SRC2..."
    cp -r $WIKITEXT_SRC2/* $SLURM_TMPDIR/wikitext/
else
    echo "Downloading WikiText-103..."
    cd $SLURM_TMPDIR/wikitext
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip -q wikitext-103-raw-v1.zip
    mv wikitext-103-raw/* .
    rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
    cd $SLURM_TMPDIR
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
