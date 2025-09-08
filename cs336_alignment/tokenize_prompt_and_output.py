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
    prompt_tokenize = tokenizer(prompt_strs, return_tensors="pt", padding=True)
    output_tokenize = tokenizer(output_strs, return_tensors="pt", padding=True)
    input_ids = torch.cat([prompt_tokenize.input_ids, output_tokenize.input_ids], dim=1)
    labels = torch.tensor(input_ids.shape[1] - 1, dtype=input_ids.dtype) # everything but first token.
    input_ids = input_ids[:, :-1] # everything but last token.
    response_mask = torch.tensor(input_ids.shape[1]-1, dtype=bool) # mask only the response tokens in the labels.
    response_mask[:, len(prompt_strs)-1:] = True # the first token was cut off because response_mask is aligned with labels.
    # response_mask[:, :len(prompt_strs)-1] = False # the first token was cut off because response_mask is aligned with labels.
    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}