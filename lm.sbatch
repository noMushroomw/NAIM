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

# Copy WikiText-103 to NVMe
echo "Copying WikiText-103 to NVMe..."
mkdir -p $SLURM_TMPDIR/wikitext
if [ -d "/blue/wdixon/wang.yixuan/data/wikitext-103" ]; then
    cp -r /blue/wdixon/wang.yixuan/data/wikitext-103/* $SLURM_TMPDIR/wikitext/
elif [ -d "/blue/wdixon/wang.yixuan/data/wikitext-103-raw" ]; then
    cp -r /blue/wdixon/wang.yixuan/data/wikitext-103-raw/* $SLURM_TMPDIR/wikitext/
else
    echo "Downloading WikiText-103..."
    cd $SLURM_TMPDIR/wikitext
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip -q wikitext-103-raw-v1.zip
    mv wikitext-103-raw/* .
    rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
fi

echo "Data files:"
ls -la $SLURM_TMPDIR/wikitext/

# Copy scripts
cp $BASE/*.py $SLURM_TMPDIR/
cd $SLURM_TMPDIR

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Starting training at $(date)"
torchrun --nproc_per_node=8 --master_port=29501 train_lm.py \
    --data $SLURM_TMPDIR/wikitext \
    --output $BASE/lm \
    --steps 100000

echo "=== Completed at $(date) ==="
