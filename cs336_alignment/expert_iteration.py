import utils
import vllm_utils
import torch
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import argparse
import os
from drgrpo_grader import r1_zero_reward_fn
import numpy as np
import gc
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="../data/gsm8k/train.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, default="../data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output_path", type=str, default="outputs/sft_experiment.jsonl")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--num_sft_examples", type=int, default=None)
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--num_expert_iterations", type=int, default=5)
    parser.add_argument("--expert_batch_size", type=int, default=512)
    parser.add_argument("--max_tokens_train", type=int, default=640)
    parser.add_argument("--max_tokens_eval", type=int, default=1024)
    return parser.parse_args()

args = get_args()
# HARDCODED STUFF
TRAIN_DATASET_PATH = args.train_dataset_path
EVAL_DATASET_PATH = args.eval_dataset_path
PROMPT_PATH = args.prompt_path
OUTPUT_PATH = args.output_path

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LR = args.lr
NUM_EPOCHS = args.num_epochs
WANDB_PROJECT = "expert-iteration-experiment"
NUM_SFT_EXAMPLES = args.num_sft_examples
NUM_EXPERT_ITERATIONS = args.num_expert_iterations
NUM_ROLLOUTS = args.num_rollouts
EXPERT_BATCH_SIZE = args.expert_batch_size
MAX_TOKENS_TRAIN = args.max_tokens_train
MAX_TOKENS_EVAL = args.max_tokens_eval

# wandb.init(project="sft-experiment") 
# set wandb experiment name to include expert_batch_size, epochs, num_rollouts, num_expert_iterations
wandb.init(project=WANDB_PROJECT, name=f"expert_batch_size_{EXPERT_BATCH_SIZE}_epochs_{NUM_EPOCHS}_num_rollouts_{NUM_ROLLOUTS}_num_expert_iterations_{NUM_EXPERT_ITERATIONS}")
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
utils.mem("after HF policy load")

# load vllm model
print("Loading vllm model...")
vllm_model = vllm_utils.init_vllm(model_id, device_vllm, SEED)
utils.mem("after vLLM init", "cuda:0")

print("Loading model...")
print("Loading dataset...")
# load dataset
train_dataset = utils.load_dataset(TRAIN_DATASET_PATH)
shuffle_indices = torch.randperm(len(train_dataset)) # just shuffle it the first time too.
train_dataset = [train_dataset[i] for i in shuffle_indices]
if NUM_SFT_EXAMPLES is not None:
    train_dataset = train_dataset[:NUM_SFT_EXAMPLES]
eval_dataset = utils.load_dataset(EVAL_DATASET_PATH)
# OK now we're going to process the dataset into the r_1_zero format.
train_dataset_r1_zero = utils.data_set_to_prompt_response_answer(train_dataset)
# train_dataset_tokenized = utils.tokenize_prompt_and_output([data["prompt"] for data in train_dataset_r1_zero], [data["response"] for data in train_dataset_r1_zero], tokenizer)

eval_dataset_r1_zero = utils.data_set_to_prompt_response_answer(eval_dataset)
eval_dataset_tokenized = utils.tokenize_prompt_and_output([data["prompt"] for data in eval_dataset_r1_zero], [data["response"] for data in eval_dataset_r1_zero], tokenizer)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# train_input_ids = train_dataset_tokenized["input_ids"].to(device_hf)
# train_labels = train_dataset_tokenized["labels"].to(device_hf)
# train_response_mask = train_dataset_tokenized["response_mask"].to(device_hf)


# check the eval:    
# So annoying thing here is that we need to retokenizer our dataset every time we do an expert iteration.
utils.mem_reset_peak()
utils.mem("EI step start (before rollouts)")
for expert_iteration in range(NUM_EXPERT_ITERATIONS):
    shuffle_indices = torch.randperm(len(train_dataset))
    # now we're going to choose a subset of them to make our expert iteration batch
    expert_batch_indices = shuffle_indices[:EXPERT_BATCH_SIZE]
    expert_batch_r1_zero = [train_dataset_r1_zero[i] for i in expert_batch_indices]
    vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
    utils.mem("after HF policy load") 
    expert_batch = utils.make_expert_iteration_batch(vllm_model, expert_batch_r1_zero, EXPERT_BATCH_SIZE, NUM_ROLLOUTS, max_tokens=MAX_TOKENS_TRAIN)
    utils.mem("after rollouts cpu side", "cuda:0")   # shows vLLM KV cache staying on cuda:0
    utils.mem("after rollouts host processing", "cuda:1")
    print("Model.device: ", model.device)
    if len(expert_batch) == 0:
        print("No reward, skipping expert iteration.")
        continue
    expert_batch_tokenized = utils.tokenize_prompt_and_output([data["prompt"] for data in expert_batch], [data["response"] for data in expert_batch], tokenizer)
    input_ids = expert_batch_tokenized["input_ids"].to(device_hf)
    labels = expert_batch_tokenized["labels"].to(device_hf)
    response_mask = expert_batch_tokenized["response_mask"].to(device_hf)
    
    for epoch in range(NUM_EPOCHS):
        shuffle_expert_indices = torch.randperm(len(expert_batch))
        ema_loss = float("inf")
        ema_reward = float("inf")
        ema_format_reward = float("inf")
        for i in range(len(expert_batch) // BATCH_SIZE):
            print(f"Expert Iteration {expert_iteration}, Epoch {epoch}, Batch {i}/{len(expert_batch) // BATCH_SIZE}")
            print(f"EMA Loss: {ema_loss:.4f}")
            last_index = min((i+1) * BATCH_SIZE, len(expert_batch))
            batch_indices = shuffle_expert_indices[i * BATCH_SIZE:last_index]
            input_ids_batch = input_ids[batch_indices, :]
            labels_batch = labels[batch_indices, :]
            response_mask_batch = response_mask[batch_indices, :]

            # Compute the policy log probs
            policy_log_probs = utils.get_response_log_probs(model, input_ids_batch, labels_batch, return_token_entropy=False)["log_probs"]

            # Compute the loss
            normalize_constant = response_mask_batch.sum().item()
            loss, _ = utils.sft_microbatch_train_step(policy_log_probs, response_mask_batch, GRADIENT_ACCUMULATION_STEPS, normalize_constant=normalize_constant)

            if i == 0:
                ema_loss = loss.item()
            else:
                ema_loss = 0.9 * ema_loss + 0.1 * loss.item()
            wandb.log({"ema_loss": ema_loss})
            if (i+1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # gc.collect(); torch.cuda.empty_cache() # You can include this but it slows everything down a bit.
                utils.mem("after step")

    
        with torch.no_grad():
            print(f"Expert Iteration {expert_iteration}, Epoch {epoch}, Evaluating...")
            vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
            log_generations_dict = utils.log_generations(vllm_model, model, tokenizer, eval_dataset_r1_zero, batch_size=BATCH_SIZE, max_tokens=MAX_TOKENS_EVAL)
            wandb.log(log_generations_dict) # index x by epoch
            histogram = utils.count_histogram(log_generations_dict["examples"])
            print("histogram: ", histogram)
            val_accuracy = histogram["correct with both format and answer reward 1"] / sum(histogram.values())
            print("Percentage of correct examples: ", val_accuracy)
            step_count = NUM_EPOCHS*expert_iteration + epoch
            wandb.log({"val_accuracy": val_accuracy, "epoch": epoch, "expert_iteration": expert_iteration, "step_count": step_count, "expert_batch_size": EXPERT_BATCH_SIZE}) # make the x axis of plot epoch


            utils.print_format_reward_1_answer_reward_1(log_generations_dict["examples"], 3)
            utils.print_format_reward_0(log_generations_dict["examples"], 3)
            utils.print_format_reward_1_answer_reward_0(log_generations_dict["examples"], 3)
