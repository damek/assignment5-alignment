import torch
import utils
import json
from vllm import SamplingParams
from drgrpo_grader import r1_zero_reward_fn

DATASET_PATH = "../data/gsm8k/test.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-Math-1.5B"

dataset = utils.load_dataset(DATASET_PATH)
dataset_r1_zero = utils.data_set_to_prompt_response_answer(dataset)
model = utils.create_model(MODEL_NAME_OR_PATH)
eval_sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024)
eval_sampling_params.stop = ["</answer>"]
eval_sampling_params.include_stop_str_in_output = True
rewards, responses = utils.evaluate_vllm(model, r1_zero_reward_fn, dataset_r1_zero, eval_sampling_params)

outputs_positives = [i for i, reward in enumerate(rewards) if reward["reward"] == 1]
print(len(outputs_positives))

with open("../data/gsm8k/test_positives.jsonl", "w") as f:
    for i in outputs_positives:
        f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")