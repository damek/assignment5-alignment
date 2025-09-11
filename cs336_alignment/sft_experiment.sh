

# run for num_sft_examples 128, 256, 512, and None (which is all)

for NUM_SFT_EXAMPLES in 128 256 512 None; do
    uv run sft_experiment.py --num_sft_examples $NUM_SFT_EXAMPLES
done