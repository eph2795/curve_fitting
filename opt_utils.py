import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import minimize


def func(x, exp_data):
    n = exp_data.shape[0]
    return np.square(f_param(x, n) - exp_data).sum()


def f_param(x, n):
    a1, a2, a3, a, b, c = x
    r = np.arange(0, n, dtype=np.float64)
    f1 = np.exp(-r/a)
    f2 = np.exp(-r/b)
    f3 = np.where(r <= c, np.square(1 - r / c), 0)
    f_all = a1 * f1 + a2 * f2 + a3 * f3
    return f_all


def get_x0():
    x = (np.random.rand(6) + 0.1) / 1.1
    x[0:3] = x[0:3] / x[0:3].sum()
    x[3] = x[3] if x[3] > x[4] else x[4] 
    return x