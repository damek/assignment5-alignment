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
    raw_rewards = []
    for r in rollout_responses:
        reward = reward_fn(repeated_ground_truths, r)
        rewards.append(reward)
        raw_rewards.append(reward["reward"])

    raw_rewards = torch.tensor(raw_rewards)
    if normalize_by_std:
        advantages = (raw_rewards - raw_rewards.mean()) / (raw_rewards.std() + advantage_eps)
    else:
        advantages = raw_rewards - raw_rewards.mean()


    metadata = {
        "rewards_mean": raw_rewards.mean(),
        "rewards_std": raw_rewards.std(),
        "rewards_min": raw_rewards.min(),
        "rewards_max": raw_rewards.max(),
        "fine_grained_rewards": rewards,
    }        

    return advantages, raw_rewards, metadata