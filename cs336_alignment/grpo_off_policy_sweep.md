# Problem (grpo_off_policy_sweep): Off-policy GRPO hyperparameter sweep (4 points)
(12 H100 hrs)

Deliverable: Fixing rollout_batch_size = 256, choose a range over epochs_per_rollout_ ⌋
batch and train_batch_size to sweep over. First do a broad sweep for a limited number of GRPO
steps (<50) to get a sense of the performance landscape, and then a more focused sweep for a larger
number of GRPO steps (200). Provide a brief experiment log explaining the ranges you chose.
Compare to your on-policy run with epochs_per_rollout_batch = 1 and train_batch_size =
256, reporting plots with respect to number of validation steps as well as with respect to wall-clock
time.
Report the validation answer reward curves. Comment on the findings, including any other metrics
that have a noticeable trend such as entropy and response length. Compare the entropy of the model’s
responses over training to what you observed in the EI experiment.
Hint: you will need to change gradient_accumulation_steps to keep memory usage constant.

---- 

Strategy, going to run for a bit and terminate early when things start to when validation loss stalls Going to try to find the extremes: 


too small and too large epochs/train_batch_size and do some bisection (because I don't want to wait 12 hrs).

I've tried in this order:
(epoch, train_batch_size)
Parallel batch 1: 
- (1, 1024): too slow
- (5, 256): crashed out
Parallel batch 2: We're going to try to stick to the same budget per grpo_iteration as in the original experiments (256 gradients touched in total).
- (2, 128): 
    - Hypothesis: Perhaps we should split the batch smaller over multiple epochs?
- (32, 8): 
    - Hypothesis: This is perhaps the most extreme.