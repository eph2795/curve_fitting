import numpy as np

from scipy.optimize import Bounds
from scipy.optimize import minimize


def fit_curve(exp_data, f, get_x0, unpack, bounds, cons, **kwargs):
    x = get_x0()

    f_opt = lambda x: func(f, x, exp_data, **kwargs)
    res = minimize(f_opt, 
                   x, 
                   options={'ftol': 1e-9, 'disp': False},
                   method='SLSQP',
                   constraints=cons,
                   bounds=bounds)
#     params = unpack(res.x)

    results = {
        'success': bool(res['success']),
        'x0': list(x),
        'x': list(res.x),
        'SSE': f_opt(res.x),
        'exp_data': list(exp_data),
        'fitted': list(f(res.x, exp_data.shape[0], **kwargs)),
    }
#     results.update(params)
    
    return results


def func(f, x, exp_data, **kwargs):
    n = exp_data.shape[0]
    return np.square(f(x, n, **kwargs) - exp_data).sum()


def f_standard(x, n, **kwargs):
    a1, a2, a3, a, b, c = x
    r = np.arange(0, n, dtype=np.float64)
    f1 = np.exp(-r/a)
    f2 = np.exp(-r/b)
    f3 = np.where(r <= c, np.square(1 - r / c), 0)
    f_all = a1 * f1 + a2 * f2 + a3 * f3
    return f_all


def f_exp(x, n, **kwargs):
    m, R = x
    phi = kwargs['phi']
    r = np.arange(0, n, dtype=np.float64)
    f_all = np.power(phi, 2 * r * (m + 1) / (np.pi * R * (m + 2)))
    return f_all


def get_x0_standard():
    x = (np.random.rand(6) + 0.1) / 1.1
    x[0:3] = x[0:3] / x[0:3].sum()
    x[3] = x[3] if x[3] > x[4] else x[4] + 0.001 
    return x


def get_x0_exp():
    x = (np.random.rand(2) + 0.1) / 1.1
    return x


def unpack_standard(x):
    return {
        'a1': x[0],
        'a2': x[1],
        'a3': x[2],
        'a': x[3],
        'b': x[4],
        'c': x[5],
    }

def unpack_exp(x):
    return {
        'm': x[0],
        'R': x[1],
    }
