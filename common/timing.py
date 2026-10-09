"""Timing helpers. GPU stages must synchronize before/after so wall-clock
numbers reflect actual GPU work, not just kernel-launch latency."""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def timer_ms(sync_cuda: bool = False, device=None):
    """Yields a callable; call it after the `with` block to get elapsed ms.
    Usage:
        with timer_ms(sync_cuda=True) as t:
            ... gpu work ...
        elapsed = t()
    """
    if sync_cuda:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = {"ms": 0.0}

    def _get():
        return result["ms"]

    try:
        yield _get
    finally:
        if sync_cuda:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
        result["ms"] = (time.perf_counter() - start) * 1000.0
