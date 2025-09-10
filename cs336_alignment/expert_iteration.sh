# expert batch sizes 512, 2048
# num rollouts 2, 4
# num expert iterations 5
# number of epochs 1, 2

for BATCH_SIZE in 512 2048; do
    for NUM_ROLLOUTS in 2 4; do
        for NUM_EXPERT_ITERATIONS in 5; do
            for EXPERT_BATCH_SIZE in 512 2048; do
                uv run expert_iteration.py --batch_size $BATCH_SIZE --num_rollouts $NUM_ROLLOUTS --num_expert_iterations $NUM_EXPERT_ITERATIONS --expert_batch_size $EXPERT_BATCH_SIZE
            done
        done
    done
done