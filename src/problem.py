"""CEC 2017 problem wrapper around the compiled organizers' reference code."""
import ctypes, os
import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LIB = ctypes.CDLL(os.path.join(_HERE, "libcec17.so"))
_LIB.cec17_chdir(os.path.join(_HERE).encode())
_LIB.cec17_eval_batch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

VALID_FUNCS = [f for f in range(1, 31) if f != 2]  # F2 excluded per CEC2017 guidance


class CEC2017Problem:
    """One CEC 2017 function at a fixed dimension. Bounds are [-100,100]^D; f* = 100*func_num."""

    def __init__(self, func_num: int, dim: int):
        if func_num not in VALID_FUNCS:
            raise ValueError(f"func_num {func_num} not valid (F2 excluded)")
        if dim not in (10, 20, 30, 50, 100):
            raise ValueError("CEC2017 only defines D in {10,20,30,50,100}")
        self.func_num = func_num
        self.dim = dim
        self.lb = -100.0 * np.ones(dim)
        self.ub = 100.0 * np.ones(dim)
        self.f_star = 100.0 * func_num
        self.n_evals = 0

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """X: (n, dim) array -> (n,) raw objective values (NOT error)."""
        X = np.ascontiguousarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        n = X.shape[0]
        f = np.empty(n, dtype=np.float64)
        _LIB.cec17_eval_batch(
            X.ctypes.data_as(ctypes.c_void_p),
            f.ctypes.data_as(ctypes.c_void_p),
            n, self.dim, self.func_num,
        )
        self.n_evals += n
        return f

    def error(self, X: np.ndarray) -> np.ndarray:
        return self.evaluate(X) - self.f_star
