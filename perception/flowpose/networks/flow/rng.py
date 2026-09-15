import torch
import hashlib
import random
import numpy as np

def fold_in(seed: int, *args) -> int:
    """
    Simulate jax.random.fold_in via SHA256 hashing.
    Args can be anything hashable: step, rank, etc.
    """
    data = str((seed,) + args)
    h = hashlib.sha256(data.encode("utf-8")).hexdigest()
    folded_seed = int(h, 16) % (2**63)  # Safe for torch.manual_seed()
    return folded_seed

def train_step_with_rng_control(train_step_fn, model_without_ddp, step: int, base_seed: int, data=None, *args, **kwargs):
    # rank = get_rank()
    rank = 0    # we use single gpu

    seed = fold_in(base_seed, step, rank, "train_step")
    input_device = args[0].device if len(args) > 0 and torch.is_tensor(args[0]) else "cpu"

    with torch.random.fork_rng(devices=[input_device], enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available() and "cuda" in str(input_device):
            torch.cuda.manual_seed(seed)
        return train_step_fn(model_without_ddp, data, *args, **kwargs)