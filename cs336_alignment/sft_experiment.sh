

# run for num_sft_examples 128, 256, 512, and None (which is all)

for NUM_SFT_EXAMPLES in 128 256 512; do
    uv run sft_experiment.py --num_sft_examples $NUM_SFT_EXAMPLES
done

uv run sft_experiment.py

uv run filter_gsm8k_positives.py
uv run sft_experiment.py --train_dataset_path ../data/gsm8k/train_positives.jsonl --num_epochs 2