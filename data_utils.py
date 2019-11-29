import os
import json

import numpy as np
import matplotlib.pyplot as plt


def normalization(data, tag):
    exp_data = np.array(data[tag])
    phi = exp_data[0]
    if tag.startswith('S2'):
        exp_data = (exp_data - phi ** 2) / (phi - phi ** 2)
    else:
        exp_data = exp_data / phi
    return exp_data


def read_data(data_path, data_name='CFs.txt', skip_first=False):
    with open(os.path.join(data_path, data_name), 'r') as f:
        lines = f.readlines()
    data = dict()
    for line in lines[int(skip_first):]:
        vals = line.split()
        data[vals[0]] = [float(v) for v in vals[1:]]
    return data


def plot_results(result, labels=None, tag=None, dir_path=None, x_up=1, x_down=-0.1):
    if labels is None:
        labels = ['fitted', 'exp_data']
    
    plt.figure(figsize=(10, 10))
    xlim = 0
    for label in labels:
        y = result[label]
        r = np.arange(0, len(y))
        xlim = max(xlim, len(y))
        plt.plot(r, y, label=label)
    # plt.plot(r, z, label='Data S2')
#     plt.plot(r, y, label='Fitted')
    plt.legend(loc='best')
    if tag is not None:
        plt.title('Plot for {}'.format(tag))
    plt.ylim([x_down, x_up])
    plt.xlim([0, xlim])
    
    if dir_path is not None:
        plt.savefig(os.path.join(dir_path, '{}.png'.format(tag)))
        with open(os.path.join(dir_path, '{}.json'.format(tag)), 'w') as f:
            f.write(json.dumps(result, indent=2))
