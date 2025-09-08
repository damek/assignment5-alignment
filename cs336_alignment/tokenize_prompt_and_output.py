import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# def download_model_and_tokenizer():
#     model = AutoModelForCausalLM.from_pretrained(
#         "/data/a5-alignment/models/Qwen2.5-Math-1.5B",
#         torch_dtype=torch.bfloat16,
#         attn_implementation="flash_attention_2",
#     )
#     tokenizer = AutoTokenizer.from_pretrained("/data/a5-alignment/models/Qwen2.5-Math-1.5B")
#     return model, tokenizer 

# model, tokenizer = download_model_and_tokenizer()


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    prompt_tokenize = tokenizer(prompt_strs, return_tensors="pt")
    output_tokenize = tokenizer(output_strs, return_tensors="pt")
    input_ids = torch.cat([prompt_tokenize.input_ids, output_tokenize.input_ids], dim=1)
    labels = input_ids[:, 1:]# everything but first token.
    input_ids = input_ids[:, :-1] # everything but last token.
    response_mask = torch.zeros(input_ids.shape[1], dtype=bool) # mask only the response tokens in the labels.
    response_mask[prompt_strs.shape[1]-1:] = True # We're dealing with the labels and since we left off the first token, the response starts one step later

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}