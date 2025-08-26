#!/bin/bash

# -----------------------------
# Job + Environment Setup
# -----------------------------
JOB_ID=${1:-manual}                # SLURM job ID or fallback to "manual"
WANDB_MODE=online                  # 'offline' or 'online' logging
LOG_DIR="/app/logs"                # For Hydra logs
WANDB_DIR="/app/wandb"             # Bound volume for wandb artifacts
RESULT_DIR="/app/result"           # Output graphs, metrics
FIGURE_DIR="/app/figures"          # Visual outputs

# -----------------------------
# Ensure output folders exist
# -----------------------------
mkdir -p "$WANDB_DIR" "$RESULT_DIR" "$FIGURE_DIR" "$LOG_DIR"

# -----------------------------
# Logging run parameters
# -----------------------------
echo "[INFO] Starting DSAM training"
echo " - JOB_ID              : $JOB_ID"
echo " - WANDB_MODE          : $WANDB_MODE"
echo " - DATA_DIR            : $DATA_DIR"
echo " - LOG_DIR             : $LOG_DIR"
echo " - WANDB_DIR           : $WANDB_DIR"
echo " - RESULT_DIR          : $RESULT_DIR"
echo " - FIGURE_DIR          : $FIGURE_DIR"
echo " - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# -----------------------------
# Run DSAM Training
# -----------------------------
python /app/run_loso.py \
    ++log_path=$LOG_DIR \
    ++wandb.project=da-loso \
    ++wandb.dir=$WANDB_DIR \
    ++wandb.name=run-${JOB_ID} \
    ++wandb.mode=$WANDB_MODE \
    ++output.result_dir=$RESULT_DIR \
    ++output.figure_dir=$FIGURE_DIR