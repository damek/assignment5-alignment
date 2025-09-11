# expert batch sizes 512, 2048
# num rollouts 2, 4
# num expert iterations 5
# number of epochs 1, 2
# batch size 10

BATCH_SIZE=5
GRADIENT_ACCUMULATION_STEPS=4
MAX_TOKENS_TRAIN=512
for NUM_EPOCHS in 2; do
    for NUM_ROLLOUTS in 4 6; do
        for NUM_EXPERT_ITERATIONS in 5; do
            for EXPERT_BATCH_SIZE in 512 1024 2048; do
                uv run expert_iteration.py --num_rollouts $NUM_ROLLOUTS --num_expert_iterations $NUM_EXPERT_ITERATIONS --expert_batch_size $EXPERT_BATCH_SIZE --num_epochs $NUM_EPOCHS --batch_size $BATCH_SIZE --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS --max_tokens_train $MAX_TOKENS_TRAIN
            done
        done
    done
done