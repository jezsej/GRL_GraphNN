#!/bin/bash
sleep $((RANDOM % 10))  # Stagger job starts to reduce I/O load

# -----------------------------
# Job + Environment Setup
# -----------------------------
JOB_ID=${3:-manual}                # SLURM job ID or fallback to "manual"
MODEL_NAME=${1:-dsam}              # Default to 'dsam' if not provided
USE_GRL=${2:-false}                # Default to no GRL
WANDB_MODE=${4:-offline} # Set offline logging

# -----------------------------
# Scratch-space output folders
# -----------------------------
SCRATCH_BASE="${SCRATCH:-/tmp}/${USER}/da_runs/${JOB_ID}"
LOG_DIR="${SCRATCH_BASE}/logs"
WANDB_DIR="${SCRATCH_BASE}/wandb"
RESULT_DIR="${SCRATCH_BASE}/result"
FIGURE_DIR="${SCRATCH_BASE}/figures"

mkdir -p "$WANDB_DIR" "$RESULT_DIR" "$FIGURE_DIR" "$LOG_DIR"

# -----------------------------
# Detect Hydra model config group
# -----------------------------
case "$MODEL_NAME" in
  SpatioTemporalModel) MODEL_GROUP=dsam ;;
  BrainGNN) MODEL_GROUP=braingnn ;;
  BrainNetworkTransformer) MODEL_GROUP=bnt ;;
  *) echo "[ERROR] Unknown MODEL_NAME: $MODEL_NAME" && exit 1 ;;
esac

# -----------------------------
# Logging run parameters
# -----------------------------
echo "[INFO] Starting model training"
echo " - MODEL_NAME          : $MODEL_NAME"
echo " - MODEL_GROUP         : $MODEL_GROUP"
echo " - USE_GRL             : $USE_GRL"
echo " - JOB_ID              : $JOB_ID"
echo " - WANDB_MODE          : $WANDB_MODE"
echo " - DATA_DIR            : $DATA_DIR"
echo " - LOG_DIR             : $LOG_DIR"
echo " - WANDB_DIR           : $WANDB_DIR"
echo " - RESULT_DIR          : $RESULT_DIR"
echo " - FIGURE_DIR          : $FIGURE_DIR"
echo " - CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export WANDB_MODE=$WANDB_MODE

# -----------------------------
# Run LOSO Training
# -----------------------------
python /app/run_loso.py \
    models=$MODEL_GROUP \
    models.name="$MODEL_NAME" \
    domain_adaptation.use_grl="$USE_GRL" \
    ++log_path=$LOG_DIR \
    ++wandb.project=da-loso \
    ++wandb.dir=$WANDB_DIR \
    ++wandb.name=run-${JOB_ID} \
    ++wandb.mode=$WANDB_MODE \
    ++output.result_dir=$RESULT_DIR \
    ++output.figure_dir=$FIGURE_DIR

# -----------------------------
# Sync outputs back if offline mode was used
# -----------------------------
if [[ "$WANDB_MODE" == "offline" ]]; then
    echo "[INFO] Training complete. Syncing logs/artifacts to /app folders..."

    # Only sync back if training completed
    mkdir -p "/app/logs/${JOB_ID}" "/app/wandb/${JOB_ID}" "/app/result/${JOB_ID}" "/app/figures/${JOB_ID}"

    cp -r "${LOG_DIR}/."      "/app/logs/${JOB_ID}/"
    cp -r "${WANDB_DIR}/."    "/app/wandb/${JOB_ID}/"
    cp -r "${RESULT_DIR}/."   "/app/result/${JOB_ID}/"
    cp -r "${FIGURE_DIR}/."   "/app/figures/${JOB_ID}/"

    echo "[INFO] Running 'wandb sync' from: ${WANDB_DIR}"
    wandb sync --sync-all "$WANDB_DIR" || echo "[WARN] wandb sync failed — please check W&B install or network"
    echo "[INFO] Sync complete (or skipped if already synced)."
fi