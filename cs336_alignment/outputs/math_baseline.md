# Problem (math_baseline): 4 points

## Question (a)

Write a script to evaluate Qwen 2.5 Math 1.5B zero-shot performance on MATH. This script
should (1) load the MATH validation examples from /data/a5-alignment/MATH/validation.jsonl,
(2) format them as string prompts to the language model using the r1_zero prompt, and (3) gen-
erate outputs for each example. This script should also (4) calculate evaluation metrics and
(5) serialize the examples, model generations, and corresponding evaluation scores to disk for
analysis in subsequent problems.
It might be helpful for your implementation to include a method evaluate_vllm with arguments
similar to the following, as you will be able to reuse it later:
```python
def evaluate_vllm(
vllm_model: LLM,
reward_fn: Callable[[str, str], dict[str, float]],
prompts: List[str],
eval_sampling_params: SamplingParams
) -> None:
"""
Evaluate a language model on a list of prompts,
compute evaluation metrics, and serialize results to disk.
"""
```

Located at [../math_baseline.py](../math_baseline.py).

Deliverable: A script to evaluate baseline zero-shot MATH performance.

## Question (b)
(b) Run your evaluation script on Qwen 2.5 Math 1.5B. How many model generations fall into each
of the following categories: (1) correct with both format and answer reward 1, (2) format reward
1 and answer reward 0, (3) format reward 0 and answer reward 0? Observing at least 10 cases
where format reward is 0, do you think the issue is with the base model’s output, or the parser?
Why? What about in (at least 10) cases where format reward is 1 but answer reward is 0?
Deliverable: Commentary on the model and reward function performance, including examples
of each category.


### Running eval script output
For the test file of gsm8k ([../../data/gsm8k/test.jsonl](../../data/gsm8k/test.jsonl)), the histogram is:
- correct with both format and answer reward 1: 31
- format reward 1 and answer reward 0: 227
- format reward 0 and answer reward 0: 1061

This uses: 
- temperature: 1.0
- top_p: 1.0
- max_tokens: 1024
- stop: ["</answer>"]
- include_stop_str_in_output: True

### Format reward 0

See the generations in [math_baseline_format_0.txt](math_baseline_format_0.txt)

**Question**: Issue with model output or parser? 
- 7/10 questions had incorrect answers.
- The rest had parser/formatting issues.

### Format reward 1 and answer reward 0

See the generations in section [math_baseline_format_1_answer_0.txt](math_baseline_format_1_answer_0.txt)

**Question**: Issue with model output or parser? 
- 7/10 questions had incorrect answers.
- The rest had parser/formatting issues.

## Question (c)

(c) How well does the Qwen 2.5 Math 1.5B zero-shot baseline perform on MATH?
Deliverable: 1-2 sentences with evaluation metrics.

By the numbers in section [Question (b)](#question-b), the model achieves 31/1319 correct answers, which is 2.35% correct.

