export WANDB_MODE=online  # or 'offline' for debugging

echo "[INFO] Starting training..."
python src/main.py \
    training=loso.yaml \
    +dataset.root=/mnt/data/abide \
    +log_path=/mnt/logs/abide \
    +wandb.project=abide-loso \
    wandb.mode=$WANDB_MODE