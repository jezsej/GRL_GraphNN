#!/bin/bash

export WANDB_MODE=online  # or 'offline'

DATA_DIR="/app/data"
LOG_DIR="/app/logs"

mkdir -p "$LOG_DIR"

echo "[INFO] Starting training..."
python /app/run_loso.py \
    log_path=$LOG_DIR \
    ++wandb.project=da-loso \
    ++wandb.mode=$WANDB_MODE