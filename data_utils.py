import os

import numpy as np


def normalization(data, tag):
    exp_data = np.array(data[tag])
    phi = exp_data[0]
    if tag.startswith('S2'):
        exp_data = (exp_data - phi ** 2) / (phi - phi ** 2)
    else:
        exp_data = exp_data / phi
    return exp_data


def read_data(data_path):
    with open(os.path.join(data_path, 'CFs.txt'), 'r') as f:
        lines = f.readlines()
    data = dict()
    for line in lines[1:]:
        vals = line.split()
        data[vals[0]] = [float(v) for v in vals[1:]]
    return data