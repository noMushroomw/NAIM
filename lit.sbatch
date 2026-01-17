#!/bin/bash
#SBATCH --job-name=rlo_lit
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=144:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=williamsw2179@gmail.com

set -e
hostname; date; pwd
echo "SLURM_TMPDIR: $SLURM_TMPDIR"
df -h $SLURM_TMPDIR

module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_experiments"
SRC_DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
mkdir -p $BASE/{logs,lit}

# Copy ImageNet to local NVMe
LOCAL_DATA="$SLURM_TMPDIR/imagenet"
mkdir -p $LOCAL_DATA

echo "Copying ImageNet to local NVMe..."
cd "$SRC_DATA"
tar cf - train | (cd $LOCAL_DATA && tar xf -)
[ -d val ] && tar cf - val | (cd $LOCAL_DATA && tar xf -)
echo "Data ready!"

# Copy scripts
cp $SLURM_SUBMIT_DIR/optimizers.py $BASE/
cp $SLURM_SUBMIT_DIR/train_lit.py $BASE/

cd $BASE

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=8 --master_port=29510 train_lit.py \
    --optimizer all \
    --data $LOCAL_DATA \
    --output $BASE/lit \
    --epochs 30

echo "Done: $(date)"
