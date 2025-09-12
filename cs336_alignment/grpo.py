import torch
import cs336_alignment.utils as utils 
from einops import einsum

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
        print("Reward, GT: ", resp, gt)
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