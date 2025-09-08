import torch
import torch.nn.functional as F

# we're given logits, which means 
##
# p = softmax(logits)
# which means log(p) = logits - log(sum(exp(logits)))
## we need to allow the tensor of size [B, Vocab]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1) ## hahaha cheating
    return -torch.sum(log_p * p, dim=-1)
    