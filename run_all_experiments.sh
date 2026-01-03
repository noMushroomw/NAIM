#!/bin/bash
# =============================================================================
# RLO Complete Experiments - Following LION Paper
# =============================================================================
#
# This script runs ALL experiments from the LION paper:
#   Section 4.1: Image Classification (ResNet-50, ViT-S/16, ViT-B/16)
#   Section 4.2: Vision-Language (LiT)
#   Section 4.3: Image Generation (Diffusion)
#   Section 4.4: Language Modeling (GPT)
#   Ablation Studies
#
# Usage:
#   ./run_all_experiments.sh
#
# Or run individual experiments:
#   ./run_all_experiments.sh classification
#   ./run_all_experiments.sh lit
#   ./run_all_experiments.sh diffusion
#   ./run_all_experiments.sh lm
#   ./run_all_experiments.sh ablation
# =============================================================================

set -e

# Configuration
DATA_PATH="/blue/wdixon/wang.yixuan/lypcdf/imagenet_folder"
LM_DATA_PATH="./wikitext-103"
NUM_GPUS=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT="./results_${TIMESTAMP}"

mkdir -p $BASE_OUTPUT

echo "=============================================="
echo "RLO Complete Experiments"
echo "=============================================="
echo "Start time: $(date)"
echo "GPUs: $NUM_GPUS"
echo "Output: $BASE_OUTPUT"
echo "=============================================="

# Function to run with torchrun
run_experiment() {
    local script=$1
    local output_dir=$2
    shift 2
    
    echo ""
    echo ">>> Running: $script"
    echo ">>> Output: $output_dir"
    echo ""
    
    mkdir -p $output_dir
    
    torchrun --nproc_per_node=$NUM_GPUS \
        --master_port=$(shuf -i 29500-29999 -n 1) \
        $script \
        --output $output_dir \
        "$@"
}

# =============================================================================
# Section 4.1: Image Classification
# =============================================================================
run_classification() {
    echo ""
    echo "######################################################"
    echo "# Section 4.1: Image Classification on ImageNet"
    echo "######################################################"
    
    # ResNet-50: 90 epochs, batch 1024
    echo ""
    echo ">>> 4.1.1 ResNet-50 (90 epochs)"
    run_experiment train_rlo.py "$BASE_OUTPUT/4.1_classification" \
        --experiment resnet50 \
        --data $DATA_PATH \
        --optimizer all
    
    # ViT-S/16: 300 epochs, batch 4096
    echo ""
    echo ">>> 4.1.2 ViT-S/16 (300 epochs)"
    run_experiment train_rlo.py "$BASE_OUTPUT/4.1_classification" \
        --experiment vit_s16 \
        --data $DATA_PATH \
        --optimizer all
    
    # ViT-B/16: 300 epochs, batch 4096
    echo ""
    echo ">>> 4.1.3 ViT-B/16 (300 epochs)"
    run_experiment train_rlo.py "$BASE_OUTPUT/4.1_classification" \
        --experiment vit_b16 \
        --data $DATA_PATH \
        --optimizer all
}

# =============================================================================
# Section 4.2: Vision-Language (LiT)
# =============================================================================
run_lit() {
    echo ""
    echo "######################################################"
    echo "# Section 4.2: Vision-Language Contrastive Learning"
    echo "######################################################"
    
    run_experiment train_lit.py "$BASE_OUTPUT/4.2_lit" \
        --data $DATA_PATH \
        --epochs 30 \
        --optimizer all
}

# =============================================================================
# Section 4.3: Image Generation (Diffusion)
# =============================================================================
run_diffusion() {
    echo ""
    echo "######################################################"
    echo "# Section 4.3: Image Generation with Diffusion Models"
    echo "######################################################"
    
    # 64x64
    echo ""
    echo ">>> 4.3.1 Diffusion 64x64"
    run_experiment train_diffusion.py "$BASE_OUTPUT/4.3_diffusion" \
        --resolution 64 \
        --data $DATA_PATH \
        --epochs 100 \
        --optimizer all
    
    # 128x128
    echo ""
    echo ">>> 4.3.2 Diffusion 128x128"
    run_experiment train_diffusion.py "$BASE_OUTPUT/4.3_diffusion" \
        --resolution 128 \
        --data $DATA_PATH \
        --epochs 100 \
        --optimizer all
    
    # 256x256 (optional, very slow)
    # echo ""
    # echo ">>> 4.3.3 Diffusion 256x256"
    # run_experiment train_diffusion.py "$BASE_OUTPUT/4.3_diffusion" \
    #     --resolution 256 \
    #     --data $DATA_PATH \
    #     --epochs 100 \
    #     --optimizer all
}

# =============================================================================
# Section 4.4: Language Modeling
# =============================================================================
run_lm() {
    echo ""
    echo "######################################################"
    echo "# Section 4.4: Autoregressive Language Modeling"
    echo "######################################################"
    
    # Download WikiText-103 if not exists
    if [ ! -d "$LM_DATA_PATH" ]; then
        echo "Downloading WikiText-103..."
        mkdir -p $LM_DATA_PATH
        wget -O wikitext-103-raw-v1.zip https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
        unzip wikitext-103-raw-v1.zip -d ./
        mv wikitext-103-raw/* $LM_DATA_PATH/
        rm -rf wikitext-103-raw wikitext-103-raw-v1.zip
    fi
    
    run_experiment train_lm.py "$BASE_OUTPUT/4.4_lm" \
        --data $LM_DATA_PATH \
        --model_size small \
        --epochs 10 \
        --optimizer all
}

# =============================================================================
# Ablation Studies
# =============================================================================
run_ablation() {
    echo ""
    echo "######################################################"
    echo "# Ablation Studies"
    echo "######################################################"
    
    run_experiment ablation_studies.py "$BASE_OUTPUT/ablation" \
        --study all \
        --data $DATA_PATH
}

# =============================================================================
# Main
# =============================================================================

EXPERIMENT=${1:-all}

case $EXPERIMENT in
    classification)
        run_classification
        ;;
    lit)
        run_lit
        ;;
    diffusion)
        run_diffusion
        ;;
    lm)
        run_lm
        ;;
    ablation)
        run_ablation
        ;;
    all)
        run_classification
        run_lit
        run_diffusion
        run_lm
        run_ablation
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Usage: $0 [classification|lit|diffusion|lm|ablation|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "End time: $(date)"
echo "Results saved to: $BASE_OUTPUT"
echo "=============================================="
