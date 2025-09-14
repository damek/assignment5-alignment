# Problem (think_about_length_normalization): Think about length normalization (1
point)

```python
from your_utils import masked_mean, masked_normalize
ratio = torch.tensor([
[1, 1, 1, 1, 1, 1, 1,],
[1, 1, 1, 1, 1, 1, 1,],
], requires_grad=True)
advs = torch.tensor([
[2, 2, 2, 2, 2, 2, 2,],
[2, 2, 2, 2, 2, 2, 2,],
])
masks = torch.tensor([
# generation 1: 4 tokens
[1, 1, 1, 1, 0, 0, 0,],
# generation 2: 7 tokens
[1, 1, 1, 1, 1, 1, 1,],
])
# Normalize with each approach
max_gen_len = 7
masked_mean_result = masked_mean(ratio * advs, masks, dim=1)
masked_normalize_result = masked_normalize(
ratio * advs, masks, dim=1, constant_normalizer=max_gen_len)
print("masked_mean", masked_mean_result)
print("masked_normalize", masked_normalize_result)
# masked_mean tensor([2., 2.], grad_fn=<DivBackward0>)
# masked_normalize tensor([1.1429, 2.0000], grad_fn=<DivBackward0>)
masked_mean_result.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad:
# tensor([[0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])
ratio.grad.zero_()
masked_normalize_result.mean().backward()
print("ratio.grad", ratio.grad)
# ratio.grad:
# tensor([[0.1429, 0.1429, 0.1429, 0.1429, 0.0000, 0.0000, 0.0000],
# [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429]])
```

Deliverable: Compare the two approaches (without running experiments yet). What are the pros
and cons of each approach? Are there any specific settings or examples where one approach seems
better?

Masked normalize: 
1. Think of the loss function. Example $i$ token $t$, has loss $L_{i,t}/M$, where M is the max length. When example $i$ contains relatively few tokens in the loss, the penalty is small because $M$ is large. On the other hand, when $i$ contains many tokens, the penalty is larger. Thus, we expect to be penalized more for longer generations. 

Masked mean:
2. Again think of the loss function. Here, we the loss is an average over the response length. Thus, we do not favor longer or shorter generations. Anything will do.