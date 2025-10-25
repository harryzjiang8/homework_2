import numpy as np
from typing import Optional


class LinearSolver:
    """your docstring here"""

    def __init__(self, A: np.ndarray, y: np.ndarray) -> None:
        return NotImplemented

    def rls(self, reg: Optional[float] = 1.0) -> np.ndarray:
        """your docstring here"""
        return NotImplemented

    def sgd(self,
            reg: float = 1.0,
            max_iter: int = 100,
            batch_size: int = 2,
            step_size: float = 1e-2) -> np.ndarray:
        """your docstring here"""
        return NotImplemented
