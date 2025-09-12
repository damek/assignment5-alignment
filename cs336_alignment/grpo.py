import torch
import cs336_alignment.utils as utils 
from einops import einsum
from typing import Literal
from vllm import LLM, SamplingParams

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
    }        

    return advantages, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:

    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    importance_ratios = torch.exp(policy_log_probs - old_log_probs)
    term_1 = importance_ratios * advantages
    term_2 = torch.clamp(importance_ratios, 1 - cliprange, 1 + cliprange) * advantages
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
        )
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs,
        )
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
    loss = masked_mean(loss, response_mask)
    loss /= max(1, gradient_accumulation_steps)
    loss.backward()
    if "loss" in metadata:
        raise ValueError("Loss already in metadata")
    metadata["loss"] = loss
    return loss, metadata

def sample_rollouts(
    vllm_model,
    dataset: list[dict],
    num_rollouts: int,
    reward_fn,
    max_tokens = 1024,
    temperature = 1.0,
    top_p = 1.0,
) -> list[str]:
    sampling_params = SamplingParams(n=num_rollouts, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True
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
