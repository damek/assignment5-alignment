import torch
import utils 
import drgrpo_grader


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    rewards = reward_fn(rollout_responses, repeated_ground_truths)

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