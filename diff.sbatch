#!/bin/bash
#SBATCH --job-name=rlo_diff
#SBATCH --output=logs/diff_%j.out
#SBATCH --error=logs/diff_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gpus=8
#SBATCH --partition=gpu
#SBATCH --time=48:00:00
#SBATCH --account=wdixon
#SBATCH --qos=wdixon

# =============================================================================
# Section 4.3: Diffusion Models (64x64, 128x128)
# =============================================================================
# Log locations:
#   SLURM: logs/diff_<JOB_ID>.out, logs/diff_<JOB_ID>.err
#   Training: results_diffusion/diffusion_*_rank0.log
# =============================================================================

cd $SLURM_SUBMIT_DIR
mkdir -p logs results_diffusion

source ~/.bashrc
conda activate ly

# Fix dill
python << 'PATCH'
import site, os
patch = "try:\n    import dill\n    if not hasattr(dill, 'extend'):\n        dill.extend = lambda use_dill=True: None\nexcept ImportError:\n    pass\n"
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f: f.write(patch)
PATCH

echo "=============================================="
echo "Job: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================="

DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
OUT="./results_diffusion"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

echo ">>> Diffusion 64x64 (100 epochs)"
torchrun --nproc_per_node=8 --master_port=29520 \
    train_diffusion.py --resolution 64 --optimizer all --data $DATA --output $OUT --epochs 100

echo ">>> Diffusion 128x128 (100 epochs)"
torchrun --nproc_per_node=8 --master_port=29521 \
    train_diffusion.py --resolution 128 --optimizer all --data $DATA --output $OUT --epochs 100

echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
