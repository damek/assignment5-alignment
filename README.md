# CS336 Spring 2025 Assignment 5: Alignment


## How to run

```bash 
runai submit cs336-dev \ -p <user> \  -i nvcr.io/nvidia/pytorch:25.06-py3 \  -g 1 --interactive --attach \  --command -- bash # replace -g 1 with -g 4 for 4 GPUs.
git clone https://github.com/damek/assignment5-alignment.git
pip install uv
cd assignment5-alignment
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv venv
source .venv/bin/activate
uv sync
```

## Description

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment5_alignment.pdf](./cs336_spring2025_assignment5_alignment.pdf)

We include a supplemental (and completely optional) assignment on safety alignment, instruction tuning, and RLHF at [cs336_spring2025_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2025_assignment5_supplement_safety_rlhf.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

As in previous assignments, we use `uv` to manage dependencies.

1. Install all packages except `flash-attn`, then all packages (`flash-attn` is weird)
```
uv sync --no-install-package flash-attn
uv sync
```

2. Run unit tests:

``` sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

