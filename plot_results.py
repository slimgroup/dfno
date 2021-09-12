import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from collections import defaultdict
from matplotlib import cm
from pprint import pprint

parser = ArgumentParser()
parser.add_argument('--input', '-i', type=str, nargs='+', help='Files containing timing data')
parser.add_argument('--output', '-o', type=str, default='fig.png', help='Output figure image file')

args = parser.parse_args()

tree = lambda: defaultdict(tree)

def parse_data(fpath):

    data = tree()
    
    with open(fpath, 'r') as f:

        input_shape_string = f.readline()[1:-2].strip()
        input_partition_shape_string = f.readline()[1:-2].strip()

        data['input_shape'] = np.array([int(x) for x in input_shape_string.split()], dtype=int)
        data['input_partition_shape'] = np.array([int(x) for x in input_partition_shape_string.split(' ')], dtype=int)
        data['num_workers'] = np.prod(data['input_partition_shape'])

        for l in f.readlines():

            section, start, stop, diff, batch = l.strip().split(',')
            diff = float(diff)
            batch = int(batch)
            
            if section not in data[batch]:
                data[batch][section] = []

            data[batch][section].append(diff)

    return data

def get_timings(data):

    timings = defaultdict(list)

    for batch, v in data.items():

        if not isinstance(v, defaultdict):
            continue
        
        # First batch has allocation step, do not want to include timings
        if batch == 0:
            continue

        timings['forward'].append(np.mean(v['forward']))
        timings['adjoint'].append(np.mean(v['adjoint']))

    out = {}
    for k, v in timings.items():
        out[k] = np.mean(v)

    return out

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)

results = defaultdict(list)

for fpath in args.input:

    data = parse_data(fpath)
    timings = get_timings(data)

    results[data['num_workers']] = {
        'forward': timings['forward'],
        'adjoint': timings['adjoint']
    }

xs = []
ts_fwd = []
ts_adj = []

for x, v in sorted(results.items(), key=lambda i: i[0]):
    xs.append(x)
    ts_fwd.append(v['forward'])
    ts_adj.append(v['adjoint'])


ax.plot(xs, ts_fwd, '-o', color='C0', label='forward')
ax.plot(xs, ts_adj, '--o', color='C0', label='adjoint')
ax.legend()

plt.show()