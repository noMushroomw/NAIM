#!/bin/bash
# =============================================================================
# RLO Complete Experiments Runner
# =============================================================================
#
# Usage:
#   ./run_all.sh                    # Run all experiments
#   ./run_all.sh classification     # Section 4.1 only
#   ./run_all.sh lit                # Section 4.2 only
#   ./run_all.sh diffusion          # Section 4.3 only
#   ./run_all.sh lm                 # Section 4.4 only
#   ./run_all.sh ablation           # Ablation studies only
#
# =============================================================================

set -e

# Configuration
DATA_PATH="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
NUM_GPUS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=============================================="
echo "RLO Complete Experiments"
echo "=============================================="
echo "Start: $(date)"
echo "GPUs: $NUM_GPUS"
echo "Data: $DATA_PATH"
echo "=============================================="

# Fix dill version - NO LONGER NEEDED (patch is in scripts)
# echo ">>> Fixing dill version..."
# pip install dill==0.3.6 --break-system-packages -q 2>/dev/null || pip install dill==0.3.6 -q

# Verify PyTorch
python -c "import torch; print(f'PyTorch {torch.__version__} OK')"

# Get random port
get_port() {
    shuf -i 29500-29999 -n 1
}

# Run function
run_exp() {
    local script=$1
    local args=$2
    local output=$3
    
    mkdir -p $output
    echo ""
    echo ">>> Running: $script $args"
    echo ">>> Output: $output"
    
    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=$(get_port) \
        $script $args --output $output
}

# Main
EXPERIMENT=${1:-all}

case $EXPERIMENT in
    classification|4.1)
        echo ""
        echo "######################################################"
        echo "# Section 4.1: Image Classification"
        echo "######################################################"
        run_exp "train_classification.py" "--model resnet50 --data $DATA_PATH" "./results_classification"
        run_exp "train_classification.py" "--model vit_s16 --data $DATA_PATH" "./results_classification"
        run_exp "train_classification.py" "--model vit_b16 --data $DATA_PATH" "./results_classification"
        ;;
    
    lit|4.2)
        echo ""
        echo "######################################################"
        echo "# Section 4.2: Vision-Language (LiT)"
        echo "######################################################"
        run_exp "train_lit.py" "--data $DATA_PATH --epochs 30" "./results_lit"
        ;;
    
    diffusion|4.3)
        echo ""
        echo "######################################################"
        echo "# Section 4.3: Diffusion Models"
        echo "######################################################"
        run_exp "train_diffusion.py" "--resolution 64 --data $DATA_PATH --epochs 100" "./results_diffusion"
        run_exp "train_diffusion.py" "--resolution 128 --data $DATA_PATH --epochs 100" "./results_diffusion"
        ;;
    
    lm|4.4)
        echo ""
        echo "######################################################"
        echo "# Section 4.4: Language Modeling"
        echo "######################################################"
        # Download WikiText-103 if needed
        if [ ! -d "./wikitext-103" ]; then
            echo ">>> Downloading WikiText-103..."
            wget -q https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
            unzip -q wikitext-103-raw-v1.zip
            mv wikitext-103-raw wikitext-103
            rm wikitext-103-raw-v1.zip
        fi
        run_exp "train_lm.py" "--data ./wikitext-103 --epochs 10" "./results_lm"
        ;;
    
    ablation)
        echo ""
        echo "######################################################"
        echo "# Ablation Studies"
        echo "######################################################"
        run_exp "train_ablation.py" "--study all --data $DATA_PATH" "./results_ablation"
        ;;
    
    all)
        $0 classification
        $0 lit
        $0 diffusion
        $0 lm
        $0 ablation
        ;;
    
    quick)
        # Quick test: ResNet-50 + Component ablation
        echo ""
        echo "######################################################"
        echo "# Quick Validation (ResNet-50 + Components)"
        echo "######################################################"
        run_exp "train_classification.py" "--model resnet50 --data $DATA_PATH" "./results_quick"
        run_exp "train_ablation.py" "--study components --data $DATA_PATH" "./results_quick"
        ;;
    
    *)
        echo "Unknown: $EXPERIMENT"
        echo "Usage: $0 [all|classification|lit|diffusion|lm|ablation|quick]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "COMPLETE"
echo "End: $(date)"
echo "=============================================="
