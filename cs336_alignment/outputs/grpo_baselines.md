# Problem (grpo_baselines): Effect of baselining (2 points) (2 H100 hrs)

Train a policy with reinforce_with_baseline and with no_baseline.
Deliverable: Validation reward curves associated with each loss type.
Deliverable: A brief 2 sentence discussion on any other trends you notice on other logged metrics.

```bash
uv run grpo_train_loop.py --loss_type no_baseline
```

![](figures/grpo_baselines.png)

You can see the rollouts in the [log file](./grpo_baselines_log.txt)

Read about my reasoning on twitter [here](https://x.com/damekdavis/status/1967007561007210699)