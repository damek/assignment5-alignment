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
    prompt_tokenize = tokenizer(prompt_strs, padding=False)
    output_tokenize = tokenizer(output_strs, padding=False)
    ids = [p + o for p, o in zip(prompt_tokenize["input_ids"], output_tokenize["input_ids"])]
    # now add padding using the tokenizer
    sequences_to_pad = [{"input_ids": x} for x in ids]
    padded_output = tokenizer.pad(sequences_to_pad, padding=True, return_tensors="pt")

    input_ids = padded_output["input_ids"][:, :-1]
    labels = padded_output["input_ids"][:, 1:]

    p_lens = [len(x) for x in prompt_tokenize["input_ids"]]
    o_lens = [len(x) for x in output_tokenize["input_ids"]]

    response_mask = torch.zeros(input_ids.shape, dtype=bool) # mask only the response tokens in the labels.
    response_mask[p_lens-1:] = True # We're dealing with the labels and since we left off the first token, the response starts one step later

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}