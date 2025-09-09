import utils
import vllm_utils
import torch
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import argparse
import os
from drgrpo_grader import r1_zero_reward_fn

# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# HARDCODED STUFF
TRAIN_DATASET_PATH = "../data/gsm8k/train.jsonl"
EVAL_DATASET_PATH = "../data/gsm8k/test.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"
OUTPUT_PATH = "outputs/sft_experiment.jsonl"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="../data/gsm8k/train.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, default="../data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output_path", type=str, default="outputs/sft_experiment.jsonl")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--print_reward_every", type=int, default=10)
    parser.add_argument("--nb_sft_examples", type=int, default=None)
    return parser.parse_args()
## later put these into argparse.

args = get_args()

SEED = 42
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LR = args.lr
# WARMUP_STEPS = args.warmup_steps
NUM_EPOCHS = args.num_epochs
WANDB_PROJECT = "sft-experiment"
PRINT_REWARD_EVERY = args.print_reward_every
NB_SFT_EXAMPLES = args.nb_sft_examples

wandb.init(project="sft-experiment") 
# Setup wandb metrics
wandb.define_metric("train_step") # the x‑axis for training
wandb.define_metric("eval_step") # the x‑axis for evaluation
# everything that starts with train/ is tied to train_step
wandb.define_metric("train/*", step_metric="train_step")
# everything that starts with eval/ is tied to eval_step
wandb.define_metric("eval/*", step_metric="eval_step")
# set up wandb

model_id = "Qwen/Qwen2.5-Math-1.5B"
device_vllm = "cuda:0"
device_hf = "cuda:1"

# load hf model 
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device_hf)
print("Model Device: ", model.device)

# load vllm model
print("Loading vllm model...")
vllm_model = vllm_utils.init_vllm(model_id, device_vllm, SEED)

print("Loading model...")
print("Loading dataset...")
# load dataset
train_dataset = utils.load_dataset(TRAIN_DATASET_PATH)
if NB_SFT_EXAMPLES is not None:
    train_dataset = train_dataset[:NB_SFT_EXAMPLES]
eval_dataset = utils.load_dataset(EVAL_DATASET_PATH)
# OK now we're going to process the dataset into the r_1_zero format.
train_dataset_r1_zero = utils.data_set_to_prompt_response_answer(train_dataset)
train_dataset_tokenized = utils.tokenize_prompt_and_output([data["prompt"] for data in train_dataset_r1_zero], [data["response"] for data in train_dataset_r1_zero], tokenizer)

eval_dataset_r1_zero = utils.data_set_to_prompt_response_answer(eval_dataset)
eval_dataset_tokenized = utils.tokenize_prompt_and_output([data["prompt"] for data in eval_dataset_r1_zero], [data["response"] for data in eval_dataset_r1_zero], tokenizer)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

train_input_ids = train_dataset_tokenized["input_ids"].to(device_hf)
train_labels = train_dataset_tokenized["labels"].to(device_hf)
train_response_mask = train_dataset_tokenized["response_mask"].to(device_hf)

eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
# check the eval:    

for epoch in range(NUM_EPOCHS):

    # compute a random shuffle
    shuffle_indices = torch.randperm(len(train_dataset))
    ema_loss = float("inf")
    ema_reward = float("inf")
    ema_format_reward = float("inf")
    for i in range(len(train_dataset) // BATCH_SIZE):
        print(f"Epoch {epoch}, Batch {i}/{len(train_dataset) // BATCH_SIZE}")
        print(f"EMA Loss: {ema_loss:.4f}")
        # Compute a batch of training examples
        batch_indices = shuffle_indices[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        input_ids = train_input_ids[batch_indices]
        labels = train_labels[batch_indices]
        response_mask = train_response_mask[batch_indices]

        # Compute the policy log probs
        policy_log_probs = utils.get_response_log_probs(model, input_ids, labels, return_token_entropy=False)["log_probs"]

        
        # every PRINT_REWARD_EVERY steps, sample generations and print the reward from the last PRINT_REWARD_EVERY steps, then reset the rewards
        if (i+1) % PRINT_REWARD_EVERY == 0:
            print_batch = shuffle_indices[(i-PRINT_REWARD_EVERY) * BATCH_SIZE:(i) * BATCH_SIZE]
            vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
            idx = print_batch.tolist() if isinstance(print_batch, torch.Tensor) else list(print_batch)
            batch = [train_dataset_r1_zero[i] for i in idx] # batch is a list of dictionaries
            rewards, _ = utils.evaluate_vllm(vllm_model, r1_zero_reward_fn, batch, eval_sampling_params)
            if i == PRINT_REWARD_EVERY-1:
                ema_reward = torch.tensor([x["reward"] for x in rewards]).sum()/ PRINT_REWARD_EVERY/BATCH_SIZE
                ema_format_reward = torch.tensor([x["format_reward"] for x in rewards]).sum()/ PRINT_REWARD_EVERY/BATCH_SIZE
            else:
                ema_reward = 0.9 * ema_reward + 0.1 * torch.tensor([x["reward"] for x in rewards]).sum()/ PRINT_REWARD_EVERY/BATCH_SIZE
                ema_format_reward = 0.9 * ema_format_reward + 0.1 * torch.tensor([x["format_reward"] for x in rewards]).sum()/ PRINT_REWARD_EVERY/BATCH_SIZE   
            print(f"EMA Reward: {ema_reward:.4f}, EMA Format Reward: {ema_format_reward:.4f}")

        # Compute the loss
        loss, _ = utils.sft_microbatch_train_step(policy_log_probs, response_mask, GRADIENT_ACCUMULATION_STEPS)
        if i == 0:
            ema_loss = loss.item()
        else:
            ema_loss = 0.9 * ema_loss + 0.1 * loss.item()
        if (i+1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

    with torch.no_grad():
        vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
        log_generations_dict = utils.log_generations(vllm_model, model, tokenizer, eval_dataset_r1_zero)
        wandb.log(log_generations_dict)
        histogram = utils.count_histogram(log_generations_dict["examples"])
        print("histogram: ", histogram)