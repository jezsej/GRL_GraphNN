#!/bin/bash

# -----------------------------
# Job + Environment Setup
# -----------------------------
JOB_ID=${3:-manual}                # SLURM job ID or fallback to "manual"
MODEL_NAME=${1:-dsam}              # Default to 'dsam' if not provided
USE_GRL=${2:-false}                # Default to no GRL
WANDB_MODE=online                  # 'offline' or 'online' logging
LOG_DIR="/app/logs"               # For Hydra logs
WANDB_DIR="/app/wandb"            # Bound volume for wandb artifacts
RESULT_DIR="/app/result"          # Output graphs, metrics
FIGURE_DIR="/app/figures"         # Visual outputs

# -----------------------------
# Ensure output folders exist
# -----------------------------
mkdir -p "$WANDB_DIR" "$RESULT_DIR" "$FIGURE_DIR" "$LOG_DIR"

# -----------------------------
# Logging run parameters
# -----------------------------
echo "[INFO] Starting model training"
echo " - MODEL_NAME          : $MODEL_NAME"
echo " - USE_GRL             : $USE_GRL"
echo " - JOB_ID              : $JOB_ID"
echo " - WANDB_MODE          : $WANDB_MODE"
echo " - DATA_DIR            : $DATA_DIR"
echo " - LOG_DIR             : $LOG_DIR"
echo " - WANDB_DIR           : $WANDB_DIR"
echo " - RESULT_DIR          : $RESULT_DIR"
echo " - FIGURE_DIR          : $FIGURE_DIR"
echo " - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Run LOSO Training
# -----------------------------
python /app/run_loso.py \
    models.name="$MODEL_NAME" \
    domain_adaptation.use_grl="$USE_GRL" \
    ++log_path=$LOG_DIR \
    ++wandb.project=da-loso \
    ++wandb.dir=$WANDB_DIR \
    ++wandb.name=run-${JOB_ID} \
    ++wandb.mode=$WANDB_MODE \
    ++output.result_dir=$RESULT_DIR \
    ++output.figure_dir=$FIGURE_DIR