#!/bin/bash
#SBATCH --job-name=rlo_lm
#SBATCH --partition=hpg-b200
#SBATCH --gpus=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=280gb
#SBATCH --time=72:00:00
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

OUT="/blue/wdixon/wang.yixuan/rlo_results/lm"
DATA="/blue/wdixon/wang.yixuan/rlo_results/openwebtext"
mkdir -p $OUT

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

# Download OpenWebText subset if not exists
if [ ! -f "$DATA/train.txt" ]; then
    echo ">>> Preparing data..."
    mkdir -p $DATA
    # Use WikiText-103 as fallback
    cd $DATA
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip -q wikitext-103-raw-v1.zip
    mv wikitext-103-raw/wiki.train.raw train.txt
    mv wikitext-103-raw/wiki.valid.raw valid.txt
    rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
    cd -
fi

torchrun --nproc_per_node=8 --master_port=29530 train_lm.py --optimizer all --data $DATA --output $OUT --steps 100000

echo "Done: $(date)"
