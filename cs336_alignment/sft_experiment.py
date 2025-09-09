import utils
import vllm_utils
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import argparse


# HARDCODED STUFF
TRAIN_DATASET_PATH = "../data/gsm8k/test.jsonl"
EVAL_DATASET_PATH = "../data/gsm8k/test.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"
OUTPUT_PATH = "outputs/sft_experiment.jsonl"

## later put these into argparse.
SEED = 42
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 2
LR = 1e-4
WARMUP_STEPS = 100
NUM_EPOCHS = 1
WANDB_PROJECT = "sft-experiment"
WANDB_ENTITY = "cs336-assignment5"

wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
# Setup wandb metrics
wandb.define_metric("train_step") # the x‑axis for training
wandb.define_metric("eval_step") # the x‑axis for evaluation
# everything that starts with train/ is tied to train_step
wandb.define_metric("train/*", step_metric="train_step")
# everything that starts with eval/ is tied to eval_step
wandb.define_metric("eval/*", step_metric="eval_step")
# set up wandb

model_id = "Qwen/Qwen2.5-Math-1.5B"
device_vllm = "cuda:1"
device_hf = "cuda:0"

# load hf model 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device_hf)

# load vllm model
print("Loading vllm model...")
vllm_model = vllm_utils.init_vllm(model_id, device_vllm, SEED)

print("Loading dataset...")
# load dataset
train_dataset = utils.load_dataset(TRAIN_DATASET_PATH)
eval_dataset = utils.load_dataset(EVAL_DATASET_PATH)
# OK now we're going to process the dataset into the r_1_zero format.
train_dataset_r1_zero = utils.data_set_to_prompt_response_answer(train_dataset)
train_dataset_tokenized = utils.tokenize_prompt_and_output(train_dataset_r1_zero, [data["answer"] for data in train_dataset_r1_zero], tokenizer)
eval_dataset_tokenized = utils.tokenize_prompt_and_output(eval_dataset_r1_zero, [data["answer"] for data in eval_dataset_r1_zero], tokenizer)
eval_dataset_r1_zero = utils.data_set_to_prompt_response_answer(eval_dataset)

train_prompts = utils.create_prompts(train_dataset, PROMPT_PATH)
eval_prompts = utils.create_prompts(eval_dataset, PROMPT_PATH)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

train_input_ids = train_dataset_tokenized["input_ids"].to(device_hf)
train_labels = train_dataset_tokenized["labels"].to(device_hf)
train_response_mask = train_dataset_tokenized["response_mask"].to(device_hf)

for epoch in range(NUM_EPOCHS):

    # compute a random shuffle
    shuffle_indices = torch.randperm(len(train_dataset))
    for i in range(len(train_dataset) // BATCH_SIZE):
        print(f"Epoch {epoch}, Batch {i}/{len(train_dataset) // BATCH_SIZE}")
        # Compute a batch of training examples
        batch_indices = shuffle_indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        inputs_ids = train_input_ids[batch_indices]
        labels = train_labels[batch_indices]
        response_mask = train_response_mask[batch_indices]

        # Compute the policy log probs
        policy_log_probs = utils.get_response_log_probs(model, input_ids, labels, return_token_entropy=False)["log_probs"]

        # Compute the loss
        loss, _ = utils.sft_microbatch_train_step(policy_log_probs, response_mask, GRADIENT_ACCUMULATION_STEPS)
        if (i+1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

    with torch.no_grad():
        vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
        log_generations_dict = utils.log_generations(vllm_model, model, tokenizer, eval_dataset_r1_zero)
        wandb.log(log_generations_dict)
        histogram = utils.count_histogram(log_generations_dict["examples"])
        print("histogram: ", histogram)