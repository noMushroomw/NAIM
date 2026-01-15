#!/bin/bash
#SBATCH --job-name=rlo_lm
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=24:00:00
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
mkdir -p $BASE/{logs,lm}

# ============================================================
# Prepare WikiText data on LOCAL NVMe
# ============================================================
LOCAL_DATA="$SLURM_TMPDIR/wikitext"
mkdir -p $LOCAL_DATA

# Check if WikiText exists on /blue, if not download
SRC_WIKI="$BASE/data/wikitext"
if [ ! -f "$SRC_WIKI/train.txt" ]; then
    echo "Downloading WikiText-103..."
    mkdir -p $SRC_WIKI
    cd $SRC_WIKI
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip -q wikitext-103-raw-v1.zip
    mv wikitext-103-raw/wiki.train.raw train.txt
    mv wikitext-103-raw/wiki.valid.raw valid.txt
    rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
fi

# Copy to local (WikiText is small, ~500MB)
echo "Copying WikiText to local NVMe..."
cp "$SRC_WIKI/train.txt" "$LOCAL_DATA/"
cp "$SRC_WIKI/valid.txt" "$LOCAL_DATA/" 2>/dev/null || true
echo "Data ready!"

# Copy scripts
cp $SLURM_SUBMIT_DIR/optimizers.py $BASE/
cp $SLURM_SUBMIT_DIR/train_lm.py $BASE/

cd $BASE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 --master_port=29530 train_lm.py \
    --optimizer all \
    --data $LOCAL_DATA \
    --output $BASE/lm \
    --steps 50000

echo "Done: $(date)"
