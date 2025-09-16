import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
from vllm import LLM, SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import gc

PROMPT_PATH = "prompts/r1_zero.prompt"

# the way the gsm8k dataset is format, the final part of the answer has a ### at the end followed by GT. 
# Thanks to gpt5 for writing this one.

def mem(tag, device="cuda:1", enabled=False):
    if enabled:
        torch.cuda.synchronize(device)
        alloc = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[MEM {device}][{tag}] alloc={alloc:.2f} GB  reserved={reserved:.2f} GB  peak={peak:.2f} GB")

def mem_reset_peak(device="cuda:1"):
    torch.cuda.reset_peak_memory_stats(device)

def data_set_to_prompt_response_answer(records):
    """
    records: iterable of dicts like {"question": str, "answer": str}
    returns: list of {"prompt": str, "response": str, "answer": str}
    """
    out = []
    for ex in records:
        q = ex["question"].strip()
        a = ex["answer"].rstrip()
        m = re.search(r"(?:^|\n)####\s*(.+)\s*$", a)  # grab the final '#### answer'
        if m:
            final_ans = m.group(1).strip()
            reasoning = a[:m.start()].strip()
        else:  # fallback if no '####'
            lines = a.splitlines()
            final_ans = lines[-1].strip()
            reasoning = "\n".join(lines[:-1]).strip()
        get_base_prompt = open(PROMPT_PATH, "r").read()
        prompt = get_base_prompt.format(question=q)
        response = f"{reasoning}</think> <answer>{final_ans}</answer>"
        out.append({"prompt": prompt, "response": response, "answer": final_ans})
    return out
    

def load_serialized_file(file_path):
    return [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]

def print_format_reward_0(rows, nb_rows=10):
    count = 0
    for row in rows:
        if row["metrics"]["format_reward"] == 0:
            print("Question: ", row["question"])
            print("Generation: ", row["generation"])
            print("Metrics: ", row["metrics"])
            print("GT Answer: ", row["gt_answer"])
            print("--------------------------------")
            count += 1
            if count >= nb_rows:
                break

def print_format_reward_1_answer_reward_0(rows, nb_rows=10):
    count = 0
    for row in rows:
        if row["metrics"]["format_reward"] == 1 and row["metrics"]["answer_reward"] == 0:
            print("Question: ", row["question"])
            print("Generation: ", row["generation"])
            print("Metrics: ", row["metrics"])
            print("GT Answer: ", row["gt_answer"])
            print("--------------------------------")
            count += 1
            if count >= nb_rows:
                break
def print_format_reward_1_answer_reward_1(rows, nb_rows=10):
    count = 0
    for row in rows:
        if row["metrics"]["format_reward"] == 1 and row["metrics"]["answer_reward"] == 1:
            print("Question: ", row["question"])
            print("Generation: ", row["generation"])
            print("Metrics: ", row["metrics"])
            print("GT Answer: ", row["gt_answer"])
            print("--------------------------------")
            count += 1
            if count >= nb_rows:
                break

def count_histogram(rows):
    histogram = {
        "correct with both format and answer reward 1": 0,
        "format reward 1 and answer reward 0": 0,
        "format reward 0 and answer reward 0": 0,
    }
    for row in rows:
        if row.get("metrics") is None:
            break
        if row["metrics"]["format_reward"] == 1 and row["metrics"]["answer_reward"] == 1:
            histogram["correct with both format and answer reward 1"] += 1
        elif row["metrics"]["format_reward"] == 1 and row["metrics"]["answer_reward"] == 0:
            histogram["format reward 1 and answer reward 0"] += 1
        elif row["metrics"]["format_reward"] == 0 and row["metrics"]["answer_reward"] == 0:
            histogram["format reward 0 and answer reward 0"] += 1
    for row in rows: 
        if row.get("format_reward") is None:
            break
        if row["format_reward"] == 1 and row["answer_reward"] == 1:
            histogram["correct with both format and answer reward 1"] += 1
        elif row["format_reward"] == 1 and row["answer_reward"] == 0:
            histogram["format reward 1 and answer reward 0"] += 1
        elif row["format_reward"] == 0 and row["answer_reward"] == 0:
            histogram["format reward 0 and answer reward 0"] += 1
    return histogram

def load_dataset(file_path):
    dataset = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data)
    return dataset

# def create_prompts(dataset, prompt_path, number_of_prompts=None):
#     prompts = []
#     get_base_prompt = open(prompt_path, "r").read()
#     if number_of_prompts is None:
#         number_of_prompts = len(dataset)
#     for data in dataset[:number_of_prompts]:    
#         prompt = get_base_prompt
#         prompt = prompt.format(question=data["question"])
#         prompts.append(prompt)
#     return prompts

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
    prompts_list = [data["prompt"] for data in dataset]
    answers_list = [data["answer"] for data in dataset]
    responses = vllm_model.generate(prompts_list, eval_sampling_params)
    rewards = []
    for i, response in enumerate(responses):
        reward_response = []
        for output in response.outputs:
            pred = output.text
            reward = reward_fn(pred, answers_list[i])
            reward_response.append(reward)
        rewards.append(reward_response)
    return rewards, responses


# This is assuming we've done 1 rollout per example.
def serialize_to_disk(dataset, responses, rewards, eval_sampling_params, output_path, add_to_existing_file=False):
    if add_to_existing_file:
        write_mode = "a"
    else:
        write_mode = "w"
    with open(output_path, write_mode, encoding="utf-8") as f:
        for i, (ex, out, score) in enumerate(zip(dataset, responses, rewards)):
            rec = {
                "id": i,
                "question": ex["prompt"],
                "gt_raw_answer": ex["response"],
                "gt_answer": ex["answer"],
                "generation": out.outputs[0].text,
                "metrics": score[0],  
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")    

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer): 
    prompt_tokenize = tokenizer(prompt_strs, padding=False, add_special_tokens=False)
    output_tokenize = tokenizer(output_strs, padding=False, add_special_tokens=False)
    ids = [p + o for p, o in zip(prompt_tokenize["input_ids"], output_tokenize["input_ids"])]
    # now add padding using the tokenizer
    sequences_to_pad = [{"input_ids": x} for x in ids]
    padded_output = tokenizer.pad(sequences_to_pad, padding=True, return_tensors="pt")

    input_ids = padded_output["input_ids"][:, :-1]
    labels = padded_output["input_ids"][:, 1:]

    p_lens = torch.tensor([len(x)-1 for x in prompt_tokenize["input_ids"]], dtype=torch.long, device=labels.device).unsqueeze(1)
    indices = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    response_mask = indices >= p_lens

    # This line is very important. You need to also mask out the padded posistions. So you'll need to and the response mask with the attention mask.
    label_attn = padded_output["attention_mask"][:, 1:].bool()
    response_mask = response_mask & label_attn

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}
# we're given logits, which means 
##
# p = softmax(logits)
# which means log(p) = logits - log(sum(exp(logits)))
## we need to allow the tensor of size [..., Vocab]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1) ## hahaha cheating
    return -torch.sum(log_p * p, dim=-1)


def get_response_log_probs(
    model,
    input_ids,
    labels,
    return_token_entropy,
    ) -> dict[str, torch.Tensor]:
    mem("before logprob gather")
    log_prob = F.log_softmax(model(input_ids).logits, dim=-1)
    log_probs = log_prob.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mem("after logprob gather")
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    else:
        return {"log_probs": log_probs}

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
    ) -> torch.Tensor:

    if dim is not None:
        return torch.sum(tensor * mask, dim=dim) / normalize_constant
    else:
        return torch.sum(tensor * mask) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # loss = F.nll_loss(policy_log_probs, reduction="none") # nice learned about reduction
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    loss /= max(1, gradient_accumulation_steps)
    loss.backward()

    return loss, {"loss": loss}


# dataset has format {"prompt": str, "response": str, "answer": str}
def log_generations(
    vllm_model,
    hf_model,
    tokenizer,
    dataset,
    batch_size=None,
    max_tokens=1024,
    temperature=1.0,
    top_p=1.0,
    log_token_entropy=False,
    reward_fn=r1_zero_reward_fn,
):
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn
    with torch.inference_mode():
        if batch_size is None:
            batch_size = len(dataset)
        # Create prompts from dataset
        prompts = [data["prompt"] for data in dataset]
        gt_answers = [data["answer"] for data in dataset]

        # Sample generations from VLLM Model
        eval_sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        eval_sampling_params.stop = ["</answer>"]
        eval_sampling_params.include_stop_str_in_output = True
        rewards, responses = evaluate_vllm(vllm_model, r1_zero_reward_fn, dataset, eval_sampling_params)
        responses = [response.outputs[0].text for response in responses]

        # Tokenize prompt and output
        tokenized_dict = tokenize_prompt_and_output(prompts, responses, tokenizer)
        input_ids, labels, response_mask = tokenized_dict["input_ids"], tokenized_dict["labels"], tokenized_dict["response_mask"]

        # Compute token entropy
        response_length = response_mask.sum(dim=-1)
        avg_entropy = torch.zeros(input_ids.shape[0], device=input_ids.device)
        if log_token_entropy:
            input_ids = input_ids.to(hf_model.device)
            response_mask = response_mask.to(hf_model.device)
            for i in range(0, len(input_ids), batch_size):
                print(f"Computing token entropy for batch {i}/{len(input_ids)}")
                input_ids_batch = input_ids[i:i+batch_size, :]
                response_mask_batch = response_mask[i:i+batch_size, :]
                logits=hf_model(input_ids_batch).logits
                token_entropy = compute_entropy(logits)
                avg_entropy[i:i+batch_size] = ((token_entropy*response_mask_batch)).sum(dim=-1)/response_length[i:i+batch_size]
        else:
            avg_entropy.fill_(float("nan"))

        # Compute response length
        # response_length = (response_length / response_length).tolist()
        # Compute which samples are correct
        is_correct = [int(r[0]["reward"] == 1) for r in rewards]
        L = torch.tensor(response_length, dtype=torch.float32)
        C = torch.tensor(is_correct)
        # compute stats
        avg_len = L.mean().item()
        correct_response_length = (L[C]).mean().item() if C.any() else float("nan")
        incorrect_response_length = (L[~C]).mean().item() if (~C).any() else float("nan")

        # Create output dictionary
        out = []
        for p, rtxt, gt, rew, ent, ln in zip(prompts, responses, gt_answers, rewards, avg_entropy, response_length):
            out.append({
                "question": p,
                "generation": rtxt,
                "gt_answer": gt,
                "metrics": rew[0],  
                "avg_token_entropy": float(ent),
                "response_length": float(ln),
            })
        return {
            "examples": out,
            "averages": {
                "avg_response_length": float(avg_len),
                "avg_response_length_correct": float(correct_response_length),
                "avg_response_length_incorrect": float(incorrect_response_length),
            },
        }


# Most of this file assumes we do 1 rollout. This function breaks from that. Thus, there could be some backwards compatibility issues.
def make_expert_iteration_batch(
    vllm_model, 
    data_batch,
    batch_size,
    num_rollouts,
    max_tokens = 1024,
    temperature =1.0,
    top_p =1.0,
    reward_fn=r1_zero_reward_fn,
    ) -> list[dict]:
    if reward_fn is None:
        reward_fn = r1_zero_reward_fn
    prompts = [data["prompt"] for data in data_batch]
    gt_answers = [data["answer"] for data in data_batch]
    eval_sampling_params = SamplingParams(n=num_rollouts, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    eval_sampling_params.stop = ["</answer>"]
    eval_sampling_params.min_tokens = 4
    eval_sampling_params.include_stop_str_in_output = True
    rewards, responses = evaluate_vllm(vllm_model, r1_zero_reward_fn, data_batch, eval_sampling_params)
    
    # now we'll filter through the responses and only keep the correct ones, 
    # saving each one as a new training sample in a dataset 
    out = []
    total_reward = 0
    for i, (prompt, gt, response, reward) in enumerate(zip(prompts, gt_answers, responses, rewards)):
        for rew, output in zip(reward, response.outputs):
            if rew["reward"] == 1:
                out.append({
                    "prompt": prompt,
                    "response": output.text,
                    "answer": gt,
                })
                total_reward += 1
    print("Length of expert batch: ", len(out))
    print("Total reward: ", total_reward)
    return out



def get_weight_norm(model):
    with torch.no_grad():
        weight_norm = torch.tensor(0.0, device=model.device)
        for param in model.parameters():
            weight_norm += param.norm()
        return weight_norm

