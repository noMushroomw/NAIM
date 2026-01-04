#!/bin/bash
#SBATCH --job-name=rlo_lm
#SBATCH --output=logs/lm_%j.out
#SBATCH --error=logs/lm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256GB
#SBATCH --gpus=8
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --account=wdixon
#SBATCH --qos=wdixon

# =============================================================================
# Section 4.4: Language Modeling (GPT)
# =============================================================================
# Log locations:
#   SLURM: logs/lm_<JOB_ID>.out, logs/lm_<JOB_ID>.err
#   Training: results_lm/lm_rank0.log
# =============================================================================

cd $SLURM_SUBMIT_DIR
mkdir -p logs results_lm

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

OUT="./results_lm"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

# Download WikiText-103 if needed
if [ ! -d "./wikitext-103" ]; then
    echo ">>> Downloading WikiText-103..."
    wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    unzip -q wikitext-103-raw-v1.zip
    mv wikitext-103-raw wikitext-103
    rm wikitext-103-raw-v1.zip
    mv wikitext-103/wiki.train.raw wikitext-103/train.txt
    mv wikitext-103/wiki.valid.raw wikitext-103/valid.txt
    mv wikitext-103/wiki.test.raw wikitext-103/test.txt
fi

echo ">>> Language Modeling (10 epochs)"
torchrun --nproc_per_node=8 --master_port=29530 \
    train_lm.py --optimizer all --data ./wikitext-103 --output $OUT --epochs 10

echo "=============================================="
echo "Done: $(date)"
echo "=============================================="
