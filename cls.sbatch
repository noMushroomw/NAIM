#!/bin/bash
#SBATCH --job-name=rlo_cls
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=144:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=williamsw2179@gmail.com
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Section 4.1: Image Classification
# Log: cat rlo_cls-<JOB_ID>.out
# Training log: cat results_classification/*_rank0.log
# =============================================================================

hostname;date;pwd

module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

# Fix dill compatibility
python << 'PATCH'
import site, os
patch = "try:\n    import dill\n    if not hasattr(dill, 'extend'):\n        dill.extend = lambda use_dill=True: None\nexcept ImportError:\n    pass\n"
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f: f.write(patch)
    print(f"Created dill patch: {path}")
PATCH

mkdir -p results_classification

DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
OUT="./results_classification"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "=============================================="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=============================================="

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
