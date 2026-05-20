"""Deterministic seeding for sampling, training, and DataLoader workers.

Call ``set_global_seed(seed)`` once at the top of every entry point that
reads the ``seed`` config key (sample generation, LoRA training, full FT).
Pass ``seed_worker`` as the DataLoader ``worker_init_fn`` so each worker
gets a deterministic-but-distinct numpy/random seed derived from the
PyTorch base seed.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic: bool = False) -> int:
    """Seed Python ``random``, NumPy, and PyTorch (CPU + all CUDA devices).

    Args:
        seed: Integer seed. Truncated mod 2**32 for libraries that require it.
        deterministic: If True, enable cuDNN deterministic mode and disable
            its autotuner. This costs throughput, so leave off unless you
            need exact step-by-step reproduction across runs.

    Returns:
        The seed that was applied (after masking).
    """
    seed = int(seed) & 0xFFFFFFFF
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(
        "Seeded random/numpy/torch with seed=%d (deterministic=%s)",
        seed,
        deterministic,
    )
    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn`` for reproducible per-worker streams."""
    base = torch.initial_seed() % (2**32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


__all__ = ["set_global_seed", "seed_worker"]
