import os
import json

import numpy as np
import matplotlib.pyplot as plt


def normalization(data, tag):
    exp_data = np.array(data[tag])
    phi = exp_data[0]
    if tag.startswith('S2') or tag.startswith('С2'):
        exp_data = (exp_data - phi ** 2) / (phi - phi ** 2)
    else:
        exp_data = exp_data / phi
    return exp_data


def read_data(data_path, skip_first=False):
    with open(os.path.join(data_path, 'CFs.txt'), 'r') as f:
        lines = f.readlines()
    data = dict()
    for line in lines[int(skip_first):]:
        vals = line.split()
        data[vals[0]] = [float(v) for v in vals[1:]]
    return data


def plot_results(result, tag, dir_path=None):
    y = result['fitted']
    d = result['exp_data']
    r = np.arange(0, len(y))
    
    plt.figure(figsize=(10, 10))
    plt.plot(r, d, label='Data')
    # plt.plot(r, z, label='Data S2')
    plt.plot(r, y, label='Fitted')
    plt.legend(loc='best')
    plt.title(r'${corr_func}_{{ {direction} }}$'.format(corr_func=tag[0], direction=tag[1:]))
    plt.ylim([-0.1, 1])
    plt.xlim([0, 250])
    
    if dir_path is not None:
        plt.savefig(os.path.join(dir_path, '{}.png'.format(tag)), dpi=400)
        with open(os.path.join(dir_path, '{}.json'.format(tag)), 'w') as f:
            f.write(json.dumps(result, indent=2))
    plt.close('all')