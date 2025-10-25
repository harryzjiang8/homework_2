import numpy as np
from typing import Optional


class LinearSolver:
    """your docstring here"""

    def __init__(self, A: np.ndarray, y: np.ndarray) -> None:
        # check dimensions of A
        if np.ndim(A) != 2:
            raise ValueError("Initialization failed because the A matrix is not a 2D numpy array.")
        self.A = A
        # check dimensions of y
        if np.ndim(y) != 1:
            print("Warning: vector is not a 1D numpy array. Reshaped to 1D numpy array automatically.")
            y = y.reshape(-1)
        self.y = y
        # check if A and y have same number of rows
        if A.shape[0] != y.shape[0]:
            raise ValueError("Mismatched number of rows. Initialization failed.")
        return None

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
