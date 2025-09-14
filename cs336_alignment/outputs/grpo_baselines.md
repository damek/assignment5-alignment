# Problem (grpo_baselines): Effect of baselining (2 points) (2 H100 hrs)

Train a policy with reinforce_with_baseline and with no_baseline.
Deliverable: Validation reward curves associated with each loss type.
Deliverable: A brief 2 sentence discussion on any other trends you notice on other logged metrics.

```bash
uv run grpo_train_loop.py --loss_type no_baseline --lr 1.5e-5
```

![](figures/grpo_baselines.png)

