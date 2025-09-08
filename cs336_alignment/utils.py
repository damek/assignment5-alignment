import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# we're given logits, which means 
##
# p = softmax(logits)
# which means log(p) = logits - log(sum(exp(logits)))
## we need to allow the tensor of size [..., Vocab]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=-1)
    log_p = F.log_softmax(logits, dim=-1) ## hahaha cheating
    return -torch.sum(log_p * p, dim=-1)
    

def get_response_log_probs(
    model,
    input_ids,
    labels,
    return_token_entropy,
    ) -> dict[str, torch.Tensor]:

    logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs = log_probs[:, labels]
    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    else:
        return {"log_probs": log_probs}