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
    prompt_tokenize = tokenizer(prompt_strs, padding=False, add_special_tokens=False)
    output_tokenize = tokenizer(output_strs, padding=False, add_special_tokens=False)
    ids = [p + o for p, o in zip(prompt_tokenize["input_ids"], output_tokenize["input_ids"])]
    # now add padding using the tokenizer
    sequences_to_pad = [{"input_ids": x} for x in ids]
    padded_output = tokenizer.pad(sequences_to_pad, padding=True, return_tensors="pt")

    input_ids = padded_output["input_ids"][:, :-1]
    labels = padded_output["input_ids"][:, 1:]

    p_lens = torch.tensor([len(x) for x in prompt_tokenize["input_ids"]], dtype=torch.long, device=labels.device).unsqueeze(1)
    indices = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    response_mask = indices >= p_lens


    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}