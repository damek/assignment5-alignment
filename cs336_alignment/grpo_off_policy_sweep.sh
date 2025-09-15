for t in "2 256" "1 128" "4 256" "2 128" "1 64" "4 128" "2 64" "1 32"; do
    set -- $t # Convert the "tuple" into the param args $1 $2...
    EPOCHS_PER_ROLLOUT_BATCH=$1
    TRAIN_BATCH_SIZE=$2
    GRADIENT_ACCUMULATION_STEPS=$((TRAIN_BATCH_SIZE/2))
    uv run grpo_train_loop.py --train_batch_size $TRAIN_BATCH_SIZE --epochs_per_rollout_batch $EPOCHS_PER_ROLLOUT_BATCH --num_grpo_iterations 40 --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --lr 1e-5 --loss_type grpo_clip --cliprange 0.2
done