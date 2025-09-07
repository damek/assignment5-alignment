# first load the data set from ../data/gsm8k/gsm8k-test.jsonl
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

with open("../data/gsm8k/test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        print(data)
        break

for data in ds:
    print(data)
    break