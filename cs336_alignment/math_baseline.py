# first load the data set from ../data/gsm8k/gsm8k-test.jsonl
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


FILE_PATH = "../data/gsm8k/train.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"


def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            print(data["question"])
            dataset.append(data)
    return dataset

def create_prompts(dataset, prompt_path, number_of_prompts):
    prompts = []
    get_base_prompt = open(prompt_path, "r").read()
    for data in dataset[:number_of_prompts]:    
        prompt = get_base_prompt
        prompt = prompt.format(question=data["question"])
        prompts.append(prompt)
    return prompts

load_dataset(FILE_PATH)    