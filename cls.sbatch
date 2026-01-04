#!/bin/bash
#SBATCH --job-name=rlo_cls
#SBATCH --output=logs/cls_%j.out
#SBATCH --error=logs/cls_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gpus=8
#SBATCH --partition=gpu
#SBATCH --time=72:00:00
#SBATCH --account=wdixon
#SBATCH --qos=wdixon

# =============================================================================
# Section 4.1: Image Classification (ResNet-50, ViT-S/16, ViT-B/16)
# =============================================================================
# Log locations:
#   SLURM: logs/cls_<JOB_ID>.out, logs/cls_<JOB_ID>.err
#   Training: results_classification/*_rank0.log
# =============================================================================

cd $SLURM_SUBMIT_DIR
mkdir -p logs results_classification

source ~/.bashrc
conda activate ly

# Fix dill compatibility (inline, no external script needed)
python << 'PATCH'
import site, os
patch = """
try:
    import dill
    if not hasattr(dill, 'extend'):
        dill.extend = lambda use_dill=True: None
except ImportError:
    pass
"""
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f:
        f.write(patch)
    print(f"Created dill patch: {path}")
else:
    print(f"Patch exists: {path}")
PATCH

echo "=============================================="
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total --format=csv

DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
OUT="./results_classification"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

echo ">>> ResNet-50 (90 epochs)"
torchrun --nproc_per_node=8 --master_port=29500 \
    train_classification.py --model resnet50 --optimizer all --data $DATA --output $OUT

echo ">>> ViT-S/16 (300 epochs)"
torchrun --nproc_per_node=8 --master_port=29501 \
    train_classification.py --model vit_s16 --optimizer all --data $DATA --output $OUT

echo ">>> ViT-B/16 (300 epochs)"
torchrun --nproc_per_node=8 --master_port=29502 \
    train_classification.py --model vit_b16 --optimizer all --data $DATA --output $OUT

echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
