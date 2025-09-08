# first load the data set from ../data/gsm8k/gsm8k-test.jsonl
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn
import re
import os
from dataclasses import asdict

DATASET_PATH = "../data/gsm8k/train.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-Math-1.5B"
OUTPUT_PATH = "outputs/math_baseline.jsonl"

def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

def create_prompts(dataset, prompt_path, number_of_prompts=None):
    prompts = []
    get_base_prompt = open(prompt_path, "r").read()
    if number_of_prompts is None:
        number_of_prompts = len(dataset)
    for data in dataset[:number_of_prompts]:    
        prompt = get_base_prompt
        prompt = prompt.format(question=data["question"])
        prompts.append(prompt)
    return prompts

def create_model(model_name_or_path):
    model = LLM(model=model_name_or_path)
    return model

def generate_outputs(prompts, model):
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    return model.generate(prompts, sampling_params)


def extract_gt(ans: str) -> str:
    m = re.search(r"####\s*([^\n]+)", ans)
    return (m.group(1) if m else ans).strip()

def evaluate_vllm(
    vllm_model,
    reward_fn,
    dataset,
    eval_sampling_params,
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    prompts = create_prompts(dataset, PROMPT_PATH, len(dataset))
    responses = vllm_model.generate(prompts, eval_sampling_params)
    rewards = []
    for i, response in enumerate(responses):
        pred = response.outputs[0].text
        actual = extract_gt(dataset[i]["answer"])
        # print("Prediction: ", pred, "Actual: ", actual)
        reward = reward_fn(pred, actual)
        rewards.append(reward)
    return rewards, responses


def serialize_to_disk(dataset, responses, rewards, eval_sampling_params, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        # reformat eval sampling params
        eval_sampling_params = asdict(eval_sampling_params)
        for i, (ex, out, score) in enumerate(zip(dataset, responses, rewards)):
            rec = {
                "id": i,
                "question": ex["question"],
                "gt_raw_answer": ex["answer"],
                "gt_answer": extract_gt(ex["answer"]),
                "generation": out.outputs[0].text,
                "metrics": score,  # e.g., {"format_reward": 1, "answer_reward": 0, "reward": 0}
                "eval_sampling_params": eval_sampling_params,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")            

def count_histogram(rows):
    histogram = {
        "correct with both format and answer reward 1": 0,
        "format reward 1 and answer reward 0": 0,
        "format reward 0 and answer reward 0": 0,
    }

    for row in rows:
        if row["metrics"]["format_reward"] == 1 and row["metrics"]["answer_reward"] == 1:
            histogram["correct with both format and answer reward 1"] += 1
        elif row["metrics"]["format_reward"] == 1 and row["metrics"]["answer_reward"] == 0:
            histogram["format reward 1 and answer reward 0"] += 1
        elif row["metrics"]["format_reward"] == 0 and row["metrics"]["answer_reward"] == 0:
            histogram["format reward 0 and answer reward 0"] += 1
    return histogram

def load_serialized_file(file_path):
    return [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]

print("Loading dataset...")
dataset=load_dataset(DATASET_PATH)    
print("Creating prompts...")
prompts=create_prompts(dataset, PROMPT_PATH)
print("Creating model...")
model=create_model(MODEL_NAME_OR_PATH)
print("Evaluating...")
eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
eval_sampling_params.stop = ["</answer>"]
eval_sampling_params.include_stop_str_in_output = True
rewards, responses=evaluate_vllm(model, r1_zero_reward_fn, dataset, eval_sampling_params)
# mkdir outputs if it doesn't exist
if not os.path.exists("outputs"):
    os.makedirs("outputs")

serialize_to_disk(dataset, responses, rewards, eval_sampling_params, OUTPUT_PATH)
rows = load_serialized_file(OUTPUT_PATH)
histogram = count_histogram(rows)
print(histogram)
