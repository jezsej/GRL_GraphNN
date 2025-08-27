#!/bin/bash

JOB_ID=${1:-manual}
SITE=${2:-YALE}  # Default fallback

WANDB_MODE=online
DATA_DIR="/app/data"
LOG_DIR="/app/logs"                # For Hydra logs
WANDB_DIR="/app/wandb"
RESULT_DIR="/app/result"
FIGURE_DIR="/app/figures"

mkdir -p "$WANDB_DIR" "$RESULT_DIR" "$FIGURE_DIR"

echo "[INFO] Starting DSAM training"
echo " - JOB_ID         : $JOB_ID"
echo " - SITE           : $SITE"
echo " - WANDB_MODE     : $WANDB_MODE"
echo " - DATA_DIR       : $DATA_DIR"
echo " - WANDB_DIR      : $WANDB_DIR"
echo " - RESULT_DIR     : $RESULT_DIR"
echo " - FIGURE_DIR     : $FIGURE_DIR"
echo " - CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

python /app/run_loso_one_site.py \
    ++log_path=$LOG_DIR \
    ++wandb.project=da-loso-pl \
    ++wandb.dir=$WANDB_DIR \
    ++wandb.name=run-${JOB_ID}-${SITE} \
    ++wandb.mode=$WANDB_MODE \
    ++output.result_dir=$RESULT_DIR \
    ++output.figure_dir=$FIGURE_DIR \
    ++loso.site_name=$SITE