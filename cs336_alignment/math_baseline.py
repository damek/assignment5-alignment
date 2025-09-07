# first load the data set from ../data/gsm8k/gsm8k-test.jsonl
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams


FILE_PATH = "../data/gsm8k/train.jsonl"
PROMPT_PATH = "prompts/r1_zero.prompt"
MODEL_NAME_OR_PATH = "Qwen/Qwen2.5-Math-1.5B"

def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
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

def create_model(model_name_or_path):
    model = LLM(model=model_name_or_path)
    return model

def generate_outputs(prompts, model):
    sampling_params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"])
    return model.generate(prompts, sampling_params)

    return responses

print("Loading dataset...")
dataset=load_dataset(FILE_PATH)    
print("Creating prompts...")
prompts=create_prompts(dataset, PROMPT_PATH, 10)
print("Creating model...")
model=create_model(MODEL_NAME_OR_PATH)
print("Generating outputs...")
responses=generate_outputs(prompts, model)
# print(responses)

for output in responses:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")