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

# Don't use set -e, handle errors manually
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
WIKITEXT_SRC3="/blue/wdixon/wang.yixuan/lypcdf/wikitext-103"

if [ -d "$WIKITEXT_SRC" ]; then
    echo "Copying WikiText-103 from $WIKITEXT_SRC..."
    cp -r $WIKITEXT_SRC/* $SLURM_TMPDIR/wikitext/
elif [ -d "$WIKITEXT_SRC2" ]; then
    echo "Copying WikiText-103 from $WIKITEXT_SRC2..."
    cp -r $WIKITEXT_SRC2/* $SLURM_TMPDIR/wikitext/
elif [ -d "$WIKITEXT_SRC3" ]; then
    echo "Copying WikiText-103 from $WIKITEXT_SRC3..."
    cp -r $WIKITEXT_SRC3/* $SLURM_TMPDIR/wikitext/
else
    echo "WikiText-103 not found locally. Downloading..."
    cd $SLURM_TMPDIR/wikitext
    
    # Try multiple download methods
    if wget -q --timeout=60 https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip; then
        unzip -q wikitext-103-raw-v1.zip
        mv wikitext-103-raw/* . 2>/dev/null || true
        rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
    elif wget -q --timeout=60 https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw-v1.zip; then
        unzip -q wikitext-103-raw-v1.zip
        mv wikitext-103-raw/* . 2>/dev/null || true
        rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
    else
        echo "ERROR: Could not download WikiText-103"
        echo "Please manually download and place in $WIKITEXT_SRC or $WIKITEXT_SRC2"
        exit 1
    fi
    cd $SLURM_TMPDIR
fi

echo "WikiText files:"
ls -la $SLURM_TMPDIR/wikitext/

# Verify we have the data
if [ ! -f "$SLURM_TMPDIR/wikitext/wiki.train.raw" ] && [ ! -f "$SLURM_TMPDIR/wikitext/train.txt" ]; then
    echo "ERROR: WikiText training file not found!"
    echo "Expected wiki.train.raw or train.txt in $SLURM_TMPDIR/wikitext/"
    ls -la $SLURM_TMPDIR/wikitext/
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at $(date)"
torchrun --nproc_per_node=8 --master_port=29501 train_lm.py \
    --data $SLURM_TMPDIR/wikitext \
    --output $BASE/lm \
    --steps 100000

echo "=== Completed at $(date) ==="
