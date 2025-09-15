import torch
import cs336_alignment.utils as utils 
from einops import einsum
from typing import Literal
from vllm import LLM, SamplingParams
import utils

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    rewards = []
    raw_rewards = []
    for resp, gt in zip(rollout_responses, repeated_ground_truths):
        reward = reward_fn(resp, gt)
        rewards.append(reward)
        raw_rewards.append(reward["reward"])
        
    raw_rewards = torch.tensor(raw_rewards)
    advantages = torch.zeros_like(raw_rewards)
    for i in range(0, len(raw_rewards), group_size):
        group_rewards = raw_rewards[i:i+group_size]
        if normalize_by_std:
            advantages[i:i+group_size] = (group_rewards - group_rewards.mean()) / (group_rewards.std() + advantage_eps)
        else:
            advantages[i:i+group_size] = group_rewards - group_rewards.mean()

    metadata = {
        "rewards_mean": raw_rewards.mean(),
        "rewards_std": raw_rewards.std(),
        "rewards_min": raw_rewards.min(),
        "rewards_max": raw_rewards.max(),
        "fine_grained_rewards": rewards,
        "advantages_mean": advantages.mean(),
        "advantages_std": advantages.std(),
        "advantages_min": advantages.min(),
        "advantages_max": advantages.max(),
    }        

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:
    return -raw_rewards_or_advantages[:, None] * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # log_ratio   = torch.clamp(policy_log_probs - old_log_probs, -10, 10) #### BE SAFE HERE
    log_ratio = policy_log_probs - old_log_probs
    importance_ratios = torch.exp(log_ratio.to(torch.float64)).to(torch.float32)
    with torch.no_grad():
        if torch.abs(log_ratio).max() > 10:
            print("log_ratio: ", log_ratio.max())
            print("policy_log_probs: ", policy_log_probs.max())
            print("old_log_probs: ", old_log_probs.max())
    term_1 = advantages[:, None] * importance_ratios
    term_2 = advantages[:, None] * torch.clamp(importance_ratios, 1 - cliprange, 1 + cliprange)
    metadata = {
        "clipped_or_not": torch.clamp(importance_ratios, 1 - cliprange, 1 + cliprange) == importance_ratios,
    }
    return -torch.min(term_1, term_2), metadata

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs,
        ), {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        ), {}
    elif loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    ) -> torch.Tensor:

    if dim is not None:
        return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)
    else:
        return torch.sum(tensor * mask) / torch.sum(mask)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    max_new_tokens: int | None = None,
    use_length_normalization: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    if metadata is None:
        metadata = {}
    if use_length_normalization:
        loss = masked_mean(loss, response_mask, dim=1)
    else:
        loss = utils.masked_normalize(loss, response_mask, normalize_constant=max_new_tokens, dim=1)
    loss /= max(1, gradient_accumulation_steps)
    loss.mean().backward() 

    if "loss" in metadata:
        raise ValueError("Loss already in metadata")
    metadata["loss"] = loss
    return loss, metadata

def sample_rollouts(
    vllm_model,
    dataset: list[dict],
    group_size: int,
    reward_fn,
    max_tokens = 1024,
    temperature = 1.0,
    top_p = 1.0,
) -> list[str]:
    sampling_params = SamplingParams(
        n=group_size, 
        temperature=temperature, 
        top_p=top_p, 
        max_tokens=max_tokens)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
    sampling_params.min_tokens = 4
    rewards, responses = utils.evaluate_vllm(
        reward_fn=reward_fn,
        vllm_model=vllm_model,
        dataset=dataset,
        eval_sampling_params=sampling_params,
    )
    # now flatten the rewards and responses
    reward_flattened = []
    for reward in rewards:
        for r in reward:
            reward_flattened.append(r)
    prompt_response_answer_flattened = []
    for k, response in enumerate(responses):
        for r in response.outputs:
            prompt_response_answer_flattened.append(
                {
                    "prompt": dataset[k]["prompt"],
                    "response": r.text,
                    "answer": dataset[k]["answer"],
                }
            )    # flatten responses out so we can just return the list 
    # the structure is a response has num_rollouts outputs, each with a text field)
    return reward_flattened, prompt_response_answer_flattened


