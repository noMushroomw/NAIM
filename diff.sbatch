#!/bin/bash
#SBATCH --job-name=rlo_diff
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=96:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=williamsw2179@gmail.com
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# =============================================================================
# Section 4.3: Diffusion Models (64x64, 128x128)
# Log: cat rlo_diff-<JOB_ID>.out
# Training log: cat results_diffusion/diffusion_*_rank0.log
# =============================================================================

hostname;date;pwd

module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

# Fix dill
python << 'PATCH'
import site, os
patch = "try:\n    import dill\n    if not hasattr(dill, 'extend'):\n        dill.extend = lambda use_dill=True: None\nexcept ImportError:\n    pass\n"
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f: f.write(patch)
PATCH

mkdir -p results_diffusion

DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
OUT="./results_diffusion"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "=============================================="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=============================================="

echo ">>> Diffusion 64x64 (100 epochs)"
torchrun --nproc_per_node=8 --master_port=29520 \
    train_diffusion.py --resolution 64 --optimizer all --data $DATA --output $OUT --epochs 100

echo ">>> Diffusion 128x128 (100 epochs)"
torchrun --nproc_per_node=8 --master_port=29521 \
    train_diffusion.py --resolution 128 --optimizer all --data $DATA --output $OUT --epochs 100

echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
