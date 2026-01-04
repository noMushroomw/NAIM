#!/bin/bash
#SBATCH --job-name=rlo_lm
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

# =============================================================================
# Section 4.4: Language Modeling (GPT)
# Log: cat rlo_lm-<JOB_ID>.out
# Training log: cat results_lm/lm_rank0.log
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

mkdir -p results_lm

OUT="./results_lm"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4

echo "=============================================="
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=============================================="

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
