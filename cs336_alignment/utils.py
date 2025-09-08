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
    
    log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    if return_token_entropy:
        entropy = compute_entropy(logits)
        return {"log_probs": log_probs, "token_entropy": entropy}
    else:
        return {"log_probs": log_probs}

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
    ) -> torch.Tensor:

    if dim is not None:
        return torch.sum(tensor * mask, dim=dim) / normalize_constant
    else:
        return torch.sum(tensor * mask) / normalize_constant

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    # loss = F.nll_loss(policy_log_probs, reduction="none") # nice learned about reduction
    loss = -masked_normalize(policy_log_probs, response_mask, normalize_constant, dim=-1).mean()
    loss /= max(1, gradient_accumulation_steps)
    loss.backward()

    return loss, {"loss": loss}