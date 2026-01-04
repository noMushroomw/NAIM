#!/bin/bash
#SBATCH --job-name=rlo_abl
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
# Ablation Studies
# Log: cat rlo_abl-<JOB_ID>.out
# Training log: cat results_ablation/ablation_rank0.log
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

mkdir -p results_ablation

DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
OUT="./results_ablation"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "=============================================="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=============================================="

echo ">>> Ablation Studies"
torchrun --nproc_per_node=8 --master_port=29540 \
    train_ablation.py --study all --data $DATA --output $OUT

echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
