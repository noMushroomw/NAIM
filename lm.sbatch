#!/bin/bash
#SBATCH --job-name=rlo_lm
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
mkdir -p $BASE/{logs,lm,data/wikitext}

python << 'PATCH'
import site, os
patch = "try:\n    import dill\n    if not hasattr(dill, 'extend'):\n        dill.extend = lambda use_dill=True: None\nexcept ImportError:\n    pass\n"
path = os.path.join(site.getsitepackages()[0], 'sitecustomize.py')
if not os.path.exists(path):
    with open(path, 'w') as f: f.write(patch)
PATCH

if [ ! -f "$BASE/data/wikitext/train.txt" ]; then
    echo ">>> Downloading WikiText-103..."
    cd $BASE/data/wikitext
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip -q wikitext-103-raw-v1.zip
    mv wikitext-103-raw/wiki.train.raw train.txt
    mv wikitext-103-raw/wiki.valid.raw valid.txt
    rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
cd $SLURM_SUBMIT_DIR

torchrun --nproc_per_node=8 --master_port=29530 train_lm.py \
    --optimizer all --data $BASE/data/wikitext --output $BASE/lm --steps 100000

echo "Done: $(date)"
