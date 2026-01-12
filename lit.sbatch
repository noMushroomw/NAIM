#!/bin/bash
#SBATCH --job-name=rlo_lit
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=williamsw2179@gmail.com
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

hostname;date;pwd
module load python conda cuda/12.8
export XDG_RUNTIME_DIR="${SLURM_TMPDIR}"
conda activate ly

python << 'PATCH'
import site, os
patch = "try:\n    import dill\n    if not hasattr(dill, 'extend'):\n        dill.extend = lambda use_dill=True: None\nexcept ImportError:\n    pass\n"
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f: f.write(patch)
PATCH

OUT="/blue/wdixon/wang.yixuan/rlo_results/lit"
DATA="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
mkdir -p $OUT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=8 --master_port=29510 train_lit.py --optimizer all --data $DATA --output $OUT --epochs 30

echo "Done: $(date)"
