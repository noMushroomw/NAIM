#!/bin/bash
# Submit all RLO experiments for ICML
# Email notifications: Williamsw2179@gmail.com

echo "=============================================="
echo "RLO Experiments - ICML Submission"
echo "=============================================="
echo ""

BASE="/blue/wdixon/wang.yixuan/rlo_icml"

# Create directory structure
mkdir -p $BASE/{logs,ablation,lm,lit,diffusion,classification}

# Copy all scripts
echo "Copying scripts to $BASE..."
cp *.py $BASE/
cp *.sbatch $BASE/
chmod +x $BASE/*.sbatch

cd $BASE

echo ""
echo "Submitting jobs..."
echo ""

# 1. Ablation - fastest, run first (~8-12h)
echo "1. Batch Size Ablation (~12h)..."
JOB_ABL=$(sbatch --parsable abl.sbatch)
echo "   Job ID: $JOB_ABL"

# 2. LM - medium duration (~36-48h)
echo "2. Language Model (~48h)..."
JOB_LM=$(sbatch --parsable lm.sbatch)
echo "   Job ID: $JOB_LM"

# 3. LiT - medium duration (~24-36h)
echo "3. LiT Training (~36h)..."
JOB_LIT=$(sbatch --parsable lit.sbatch)
echo "   Job ID: $JOB_LIT"

# 4. Diffusion - longer (~72-96h)
echo "4. Diffusion (~96h)..."
JOB_DIFF=$(sbatch --parsable diff.sbatch)
echo "   Job ID: $JOB_DIFF"

# 5. ViT-B/16 - longest (~120h)
echo "5. ViT-B/16 Classification (~120h)..."
JOB_VIT=$(sbatch --parsable vit_b16.sbatch)
echo "   Job ID: $JOB_VIT"

echo ""
echo "=============================================="
echo "All jobs submitted!"
echo "=============================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel $JOB_ABL $JOB_LM $JOB_LIT $JOB_DIFF $JOB_VIT"
echo ""
echo "Expected completion times:"
echo "  - Ablation:       ~12 hours"
echo "  - LM:             ~48 hours"
echo "  - LiT:            ~36 hours"
echo "  - Diffusion:      ~96 hours"
echo "  - ViT-B/16:       ~120 hours"
echo ""
echo "Email notifications will be sent to: Williamsw2179@gmail.com"
