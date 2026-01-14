#!/bin/bash
#SBATCH --job-name=rlo_diff
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=300:00:00
#SBATCH --output=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.out
#SBATCH --error=/blue/wdixon/wang.yixuan/rlo_experiments/logs/%x-%j.err

hostname; date; pwd
module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

BASE="/blue/wdixon/wang.yixuan/rlo_experiments"
mkdir -p $BASE/{logs,diffusion}

python << 'PATCH'
import site, os
patch = "try:\n    import dill\n    if not hasattr(dill, 'extend'):\n        dill.extend = lambda use_dill=True: None\nexcept ImportError:\n    pass\n"
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f: f.write(patch)
PATCH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
cd $SLURM_SUBMIT_DIR

torchrun --nproc_per_node=8 --master_port=29520 train_diffusion.py \
    --resolution 64 --optimizer all \
    --data /blue/wdixon/wang.yixuan/lypcdf/imagenet_folder \
    --output $BASE/diffusion --epochs 100

torchrun --nproc_per_node=8 --master_port=29521 train_diffusion.py \
    --resolution 128 --optimizer all \
    --data /blue/wdixon/wang.yixuan/lypcdf/imagenet_folder \
    --output $BASE/diffusion --epochs 100

echo "Done: $(date)"
