import numpy as np
from typing import Optional


class LinearSolver:
    """
    Matrix A of shape (m,d)
    y vector shape (m,)
    """

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
        """
        Return coefficients of vector x as np.ndarry of shape (d,)

        :returns solved unregularized least-squares problem if reg is 0 or None
        :returns solved regularized least-squares problem if reg > 0
        :raises ValueError: if reg is negative
        """
        ATA = self.A.T@self.A
        ATy = self.A.T@self.y
        if reg == 0 or reg == None:
            return np.linalg.solve(ATA, ATy)
        
        if reg < 0:
            raise ValueError("Regularization parameter should be nonnegative or None.")

        if reg > 0:
            return np.linalg.solve(ATA + reg*np.identity(self.A.shape[1]), ATy)

    
    def sgd(self,
            reg: float = 1.0,
            max_iter: int = 100,
            batch_size: int = 2,
            step_size: float = 1e-2) -> np.ndarray:
        """Stochastic Gradient Descent

        raises ValueError: if reg is negative
        """

        if reg < 0:
            raise ValueError("Regularization parameter should be nonnegative or None.")
        
        m, d = self.A.shape
        x = np.zeros(d)
        b = np.random.choice(m, batch_size)
        A_b = self.A[b, :]
        y_b = self.y[b]

        for i in range(max_iter):
            x = x - step_size * (2*(A_b.T) @ (A_b@x - y_b) + 2*reg*x)
            
        
        
        return x







