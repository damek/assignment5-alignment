import torch
import cs336_alignment.utils as utils 


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    rewards = []
    for r in rollout_responses:
        reward = reward_fn(r, repeated_ground_truths)
        rewards.append(reward)

    rewards = torch.tensor(rewards)
    if normalize_by_std:
        rewards = (rewards - rewards.mean()) / (rewards.std() + advantage_eps)
    else:
        rewards = rewards - rewards.mean()

    metadata = {
        "rewards_mean": rewards.mean(),
        "rewards_std": rewards.std(),
        "rewards_min": rewards.min(),
        "rewards_max": rewards.max(),
    }        

    return rewards, metadata