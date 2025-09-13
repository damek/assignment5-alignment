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
import grpo
from typing import Literal
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset_path", type=str, default="../data/gsm8k/train.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, default="../data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_path", type=str, default="prompts/r1_zero.prompt")
    parser.add_argument("--output_path", type=str, default="outputs/sft_experiment.jsonl")
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--num_grpo_iterations", type=int, default=200)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_tokens_train", type=int, default=1024)
    parser.add_argument("--max_tokens_eval", type=int, default=1024)
    # parser.add_argument("--loss_type", type=Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], default="reinforce_with_baseline")
    parser.add_argument("--use_std_normalization", type=bool, default=True)
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
GRADIENT_ACCUMULATION_STEPS = args.gradient_accumulation_steps
LR = args.lr
WANDB_PROJECT = "cs336-grpo-experiment"
MAX_TOKENS_TRAIN = args.max_tokens_train
MAX_TOKENS_EVAL = args.max_tokens_eval


NUM_GRPO_ITERATIONS: int = args.num_grpo_iterations
ADVANTAGE_EPS: float = 1e-6
ROLLOUT_BATCH_SIZE: int = args.rollout_batch_size
GROUP_SIZE: int = args.group_size
EPOCHS_PER_ROLLOUT_BATCH: int = args.epochs_per_rollout_batch # On-policy
TRAIN_BATCH_SIZE: int = args.train_batch_size # On-policy
GRADIENT_ACCUMULATION_STEPS: int = args.gradient_accumulation_steps # microbatch size is 2, will fit on H100
TEMPERATURE: float = 1
TOP_P: float = 1
GPU_MEMORY_UTILIZATION: float = 0.75
LOSS_TYPE = "reinforce_with_baseline"
USE_STD_NORMALIZATION: bool = args.use_std_normalization

assert TRAIN_BATCH_SIZE % GRADIENT_ACCUMULATION_STEPS == 0, (
"train_batch_size must be divisible by gradient_accumulation_steps"
)
micro_train_batch_size = TRAIN_BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
print("micro_train_batch_size: ", micro_train_batch_size)
assert ROLLOUT_BATCH_SIZE % GROUP_SIZE == 0, (
"rollout_batch_size must be divisible by group_size"
)
n_prompts_per_rollout_batch = ROLLOUT_BATCH_SIZE // GROUP_SIZE
assert TRAIN_BATCH_SIZE >= GROUP_SIZE, (
"train_batch_size must be greater than or equal to group_size"
)
n_microbatches_per_rollout_batch = ROLLOUT_BATCH_SIZE // micro_train_batch_size

# set wandb experiment name to include num_grpo_steps, advantage_eps, rollout_batch_size, group_size, epochs_per_rollout_batch, train_batch_size, gradient_accumulation_steps, loss_type, use_std_normalization
wandb.init(project=WANDB_PROJECT, name=f"num_grpo_steps_{NUM_GRPO_ITERATIONS}_advantage_eps_{ADVANTAGE_EPS}_rollout_batch_size_{ROLLOUT_BATCH_SIZE}_group_size_{GROUP_SIZE}_epochs_per_rollout_batch_{EPOCHS_PER_ROLLOUT_BATCH}_train_batch_size_{TRAIN_BATCH_SIZE}_gradient_accumulation_steps_{GRADIENT_ACCUMULATION_STEPS}_loss_type_{LOSS_TYPE}_use_std_normalization_{USE_STD_NORMALIZATION}")
# wandb.init(mode="disabled")

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
vllm_model = vllm_utils.init_vllm(model_id, device_vllm, SEED, gpu_memory_utilization=GPU_MEMORY_UTILIZATION)
utils.mem("after vLLM init", "cuda:0")

print("Loading model...")
print("Loading dataset...")
# load dataset
train_dataset = utils.load_dataset(TRAIN_DATASET_PATH)
eval_dataset = utils.load_dataset(EVAL_DATASET_PATH)
# OK now we're going to process the dataset into the r_1_zero format.
train_dataset_r1_zero = utils.data_set_to_prompt_response_answer(train_dataset)

eval_dataset_r1_zero = utils.data_set_to_prompt_response_answer(eval_dataset)

optimizer = torch.optim.AdamW(
model.parameters(),
lr=LR,
weight_decay=0.0,
betas=(0.9, 0.95),
)


# check the eval:    
# So annoying thing here is that we need to retokenizer our dataset every time we do an expert iteration.
utils.mem_reset_peak()
utils.mem("EI step start (before rollouts)")



for grpo_iteration in range(NUM_GRPO_ITERATIONS):
    # first thing to do is sample TRAIN_BATCH_SIZE examples from the train dataset    
    sample = torch.randperm(len(train_dataset))[:n_prompts_per_rollout_batch]
    train_dataset_r1_zero_grpo_step = [train_dataset_r1_zero[i] for i in sample]
    # Load model into vllm
    vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
    utils.mem("after HF policy load")
    # Do rollouts from VLLM model
    rewards, prompt_response_answer_flattened  = grpo.sample_rollouts(vllm_model, train_dataset_r1_zero_grpo_step, GROUP_SIZE, r1_zero_reward_fn, MAX_TOKENS_TRAIN, TEMPERATURE, TOP_P)

    # Rollouts
    tokenize_samples = utils.tokenize_prompt_and_output([data["prompt"] for data in prompt_response_answer_flattened], [data["response"] for data in prompt_response_answer_flattened], tokenizer)
    input_ids = tokenize_samples["input_ids"].to(device_hf)
    labels = tokenize_samples["labels"].to(device_hf)
    response_mask = tokenize_samples["response_mask"].to(device_hf)

    advantages, raw_rewards, metadata = grpo.compute_group_normalized_rewards(r1_zero_reward_fn, prompt_response_answer_flattened, [data["answer"] for data in prompt_response_answer_flattened], GROUP_SIZE, ADVANTAGE_EPS, USE_STD_NORMALIZATION)
    print("advantages.shape: ", advantages.shape)
    print("raw_rewards.shape: ", raw_rewards.shape)
    # move to device
    advantages = advantages.to(device_hf)
    raw_rewards = raw_rewards.to(device_hf)
    old_log_probs = torch.empty(input_ids.shape, dtype=torch.float32).to(device_hf)
    if LOSS_TYPE == "grpo_clip":
        with torch.no_grad():
            for i in range(0, TRAIN_BATCH_SIZE // micro_train_batch_size):
                last_index = min((i+1) * micro_train_batch_size, TRAIN_BATCH_SIZE)
                batch_indices = torch.arange(i * micro_train_batch_size, last_index)
                input_ids_batch = input_ids[batch_indices, :]
                labels_batch = labels[batch_indices, :]
                response_mask_batch = response_mask[batch_indices, :]
                old_log_probs[batch_indices, :] = utils.get_response_log_probs(model, input_ids_batch, labels_batch, return_token_entropy=False)["log_probs"]
    
    histogram = utils.count_histogram(rewards)
    print("histogram: ", histogram)
    batch_accuracy = histogram["correct with both format and answer reward 1"] / sum(histogram.values())
    print("Percentage of correct examples: ", batch_accuracy)
    wandb.log({"batch_accuracy": batch_accuracy, "grpo_iteration": grpo_iteration})

    for epoch in range(EPOCHS_PER_ROLLOUT_BATCH):
        # Could shuffle here.
        for i in range(0, TRAIN_BATCH_SIZE // micro_train_batch_size):
            print("GRPO Iteration: ", grpo_iteration, "Epoch: ", epoch, "Microbatch: ", i, "/", TRAIN_BATCH_SIZE // micro_train_batch_size)
            last_index = min((i+1) * micro_train_batch_size, TRAIN_BATCH_SIZE)
            batch_indices = torch.arange(i * micro_train_batch_size, last_index)
            input_ids_batch = input_ids[batch_indices, :]
            labels_batch = labels[batch_indices, :]
            response_mask_batch = response_mask[batch_indices, :]
            policy_log_probs = utils.get_response_log_probs(model, input_ids_batch, labels_batch, return_token_entropy=False)["log_probs"]
            # print("policy_log_probs.shape: ", policy_log_probs.shape)
            # print("response_mask_batch.shape: ", response_mask_batch.shape)
            # print("raw_rewards[batch_indices].shape: ", raw_rewards[batch_indices].shape)
            # print("advantages.shape: ", advantages.shape)
            # print("advantages[batch_indices].shape: ", advantages[batch_indices].shape)
            # print("old_log_probs[batch_indices,:].shape: ", old_log_probs[batch_indices,:].shape)
            # print("batch_indices.shape: ", batch_indices.shape)
            utils.mem("Before grpo microbatch train step")
            loss, _ = grpo.grpo_microbatch_train_step(policy_log_probs, response_mask_batch, GRADIENT_ACCUMULATION_STEPS, LOSS_TYPE, raw_rewards=raw_rewards[batch_indices], advantages=advantages[batch_indices], old_log_probs=old_log_probs[batch_indices,:], cliprange=None)
            utils.mem("After grpo microbatch train step")

            ## log weights and gradient norms 
            with torch.no_grad():
                grad_norm = 0
                for param in model.parameters():
                    grad_norm += param.grad.norm()
            # wandb.log({"gradient_norms": grad_norm})
            # log the rewards and the norm of the advantages
            wandb.log({"rewards": raw_rewards[batch_indices].mean(), "advantages": advantages[batch_indices].mean(), "gradient_norms": grad_norm})
            # wandb.log({"gradient_norms": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)})

            if (i+1) % GRADIENT_ACCUMULATION_STEPS == 0:
                # weights norm
                with torch.no_grad():
                    weight_norm_before_step = 0
                    for param in model.parameters():
                        weight_norm_before_step += param.norm()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                with torch.no_grad():
                    weight_norm_after_step = 0
                    for param in model.parameters():
                        weight_norm_after_step += param.norm()
                wandb.log({"weight_norm_change": torch.linalg.norm(weight_norm_before_step - weight_norm_after_step)})
                optimizer.zero_grad(set_to_none=True)
                # gc.collect(); torch.cuda.empty_cache() # You can include this but it slows everything down a bit.
                utils.mem("after step")
    if grpo_iteration % 5 == 0:
        with torch.no_grad():
            print(f"GRPO Iteration {grpo_iteration}, Epoch {epoch}, Evaluating...")
            vllm_utils.load_policy_into_vllm_instance(model, vllm_model)
            log_generations_dict = utils.log_generations(vllm_model, model, tokenizer, eval_dataset_r1_zero, batch_size=ROLLOUT_BATCH_SIZE, max_tokens=MAX_TOKENS_EVAL)
            wandb.log(log_generations_dict) # index x by epoch
            histogram = utils.count_histogram(log_generations_dict["examples"])
            print("histogram: ", histogram)
            val_accuracy = histogram["correct with both format and answer reward 1"] / sum(histogram.values())
            print("Percentage of correct examples: ", val_accuracy)
            wandb.log({"val_accuracy": val_accuracy, "epoch": epoch, "grpo_iteration": grpo_iteration}) # make the x axis of plot epoch


            utils.print_format_reward_1_answer_reward_1(log_generations_dict["examples"], 3)
            utils.print_format_reward_0(log_generations_dict["examples"], 3)
            utils.print_format_reward_1_answer_reward_0(log_generations_dict["examples"], 3)
