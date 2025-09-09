import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
import re
import os
from utils import load_dataset, create_model, evaluate_vllm, serialize_to_disk, load_serialized_file, count_histogram, print_format_reward_0, print_format_reward_1_answer_reward_0, data_set_to_prompt_response_answer

DATASET_PATH = "../data/gsm8k/test.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-Math-1.5B"
OUTPUT_PATH = "outputs/math_baseline.jsonl"

print("Loading dataset...")
dataset=load_dataset(DATASET_PATH)    
print("Creating prompts...")
prompts=data_set_to_prompt_response_answer(dataset)
print("prompts[0]: ", prompts[0])
print("answer: ", prompts[0]["answer"])
print("Creating model...")
model=create_model(MODEL_NAME_OR_PATH)
print("Evaluating...")
eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
eval_sampling_params.stop = ["</answer>"]
eval_sampling_params.include_stop_str_in_output = True
rewards, responses=evaluate_vllm(model, r1_zero_reward_fn, prompts, eval_sampling_params)
if not os.path.exists("outputs"):
    os.makedirs("outputs")

serialize_to_disk(dataset, responses, rewards, eval_sampling_params, OUTPUT_PATH)
rows = load_serialized_file(OUTPUT_PATH)
histogram = count_histogram(rows)
print("Printing format reward 0...")
print_format_reward_0(rows, 10)
print("Printing format reward 1 and answer reward 0...")
print_format_reward_1_answer_reward_0(rows, 10)
print(histogram)
