"""
Backward-compatible entry: same training as :mod:`deep_learning.train`.

Prefer ``python train.py`` inside ``deep_learning/`` or
``python -m deep_learning.train`` from the project root.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from deep_learning.train import main  # noqa: E402

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    main()
