#!/bin/bash
#SBATCH --job-name=rlo_vit_s16
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=48:00:00
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
SRC_DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
mkdir -p $BASE/{logs,classification}

# ============================================================
# Copy ImageNet to LOCAL NVMe
# ============================================================
LOCAL_DATA="$SLURM_TMPDIR/imagenet"
mkdir -p $LOCAL_DATA

echo "========================================"
echo "Copying ImageNet to local NVMe..."
echo "========================================"

echo "Copying train set..."
cd "$SRC_DATA"
tar cf - train | (cd $LOCAL_DATA && tar xf -)
echo "Train copied!"

if [ -d "$SRC_DATA/val" ]; then
    echo "Copying val set..."
    tar cf - val | (cd $LOCAL_DATA && tar xf -)
    echo "Val copied!"
fi

echo "Data ready at: $LOCAL_DATA"
ls -la $LOCAL_DATA/

# Copy scripts
cp $SLURM_SUBMIT_DIR/optimizers.py $BASE/
cp $SLURM_SUBMIT_DIR/train_classification.py $BASE/

cd $BASE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings for stability
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

# ViT-S/16 training
torchrun --nproc_per_node=8 --master_port=29500 train_classification.py \
    --model vit_s16 \
    --optimizer all \
    --data $LOCAL_DATA \
    --output $BASE/classification \
    --epochs 90

echo "Done: $(date)"
