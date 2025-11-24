#!/bin/bash

# Phase 3 Training Script - Multi-Scale Prototype-Guided Adaptive Training
# Based on ScaleProtoSeg multi-scale prototype learning

echo "Starting FD-MSP Phase 3 Training - Multi-Scale Prototype-Guided Adaptive Training"
echo "Based on ScaleProtoSeg implementation for improved polyp segmentation"

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:./ScaleProtoSeg-main

# Training parameters
BATCH_SIZE=2
NUM_WORKERS=4
NUM_STEPS=50000
LEARNING_RATE=1e-4
RESUME_PATH="./snapshots/EndoTect2CVC-ClinicDB_phase2/GTA5_7000_0.8340479297984174.pth"
SNAPSHOT_DIR="./snapshots_phase3/"
LOG_DIR="./logs_prototype/"

# Prototype network parameters
PROTOTYPE_SHAPE="20,128,1,1"  # (num_prototypes, channels, height, width)
NUM_SCALES=4
LAMBDA_PROTO=0.1
LAMBDA_DIVERSITY=0.01
LAMBDA_SEPARATION=0.01

# Dataset parameters
SRC_DATASET="endotect"
TGT_DATASET="cvc_clinicdb"
TGT_VAL_DATASET="cvc_clinicdb_val"
SRC_ROOTPATH="datasets/EndoTect"
TGT_ROOTPATH="datasets/CVC-ClinicDB"

# Create output directories
mkdir -p $SNAPSHOT_DIR
mkdir -p $LOG_DIR

echo "Training Configuration:"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Training steps: $NUM_STEPS"
echo "  - Learning rate: $LEARNING_RATE"
echo "  - Prototype shape: $PROTOTYPE_SHAPE"
echo "  - Number of scales: $NUM_SCALES"
echo "  - Prototype loss weight: $LAMBDA_PROTO"
echo "  - Diversity loss weight: $LAMBDA_DIVERSITY"
echo "  - Separation loss weight: $LAMBDA_SEPARATION"
echo "  - Resume from: $RESUME_PATH"
echo "  - Output directory: $SNAPSHOT_DIR"

# Start training - Single GPU mode
python train_phase3.py \
    --model DeepLab \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --num-steps $NUM_STEPS \
    --learning-rate $LEARNING_RATE \
    --resume $RESUME_PATH \
    --snapshot-dir $SNAPSHOT_DIR \
    --log-dir $LOG_DIR \
    --gpus "0" \
    --src_dataset $SRC_DATASET \
    --tgt_dataset $TGT_DATASET \
    --tgt_val_dataset $TGT_VAL_DATASET \
    --src_rootpath $SRC_ROOTPATH \
    --tgt_rootpath $TGT_ROOTPATH \
    --prototype_shape $PROTOTYPE_SHAPE \
    --num_scales $NUM_SCALES \
    --lambda_proto $LAMBDA_PROTO \
    --lambda_diversity $LAMBDA_DIVERSITY \
    --lambda_separation $LAMBDA_SEPARATION \
    --layer 1 \
    --hidden_dim 128 \
    --num-classes 2 \
    --resize 384 \
    --rcrop 384,384 \
    --hflip 0.5 \
    --clrjit_params 0.5,0.5,0.5,0.2 \
    --thresholds_path high_thres.npy \
    --soft_labels_folder "" \
    --pseudo_labels_folder "" \
    --tensorboard \
    --ngpus_per_node 1 \
    --print-every 100 \
    --save-pred-every 200 \
    --freeze_bn \
    --src_loss_weight 1.0

echo "Phase 3 training completed!"
echo "Model saved in: $SNAPSHOT_DIR"
echo "Logs saved in: $LOG_DIR"

# Optional: Run validation
echo "Running final validation..."
python validate_phase3.py \
    --model_path "${SNAPSHOT_DIR}/Phase3_best.pth" \
    --dataset_path $TGT_ROOTPATH \
    --output_dir "${SNAPSHOT_DIR}/validation_results/"

echo "Validation completed! Results saved in: ${SNAPSHOT_DIR}/validation_results/"
