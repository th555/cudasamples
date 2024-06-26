import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import subprocess, os
from collections import defaultdict
from random import random
import csv

sizes = [(100,100),(1000,1000),(100,20000),(20000,100),(5000,5000)]

runtimes = []

def run(runstr):
    print(runstr)
    times = []
    for _ in range(3):
        p = subprocess.Popen(runstr, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        # (time, flops)
        times.append((float(out.split()[-2]), float(out.split()[-1])))
    return min(times)


for i, (M, N) in enumerate(sizes):
    runtimes.append([])
    base="prun -v -np 1 -1 -native '-C TitanX-Pascal --gres=gpu:1' heat_cuda -L 0 -H 100 -c ../../images/areas_{M}x{N}.pgm -t ../../images/areas_{M}x{N}.pgm -m {M} -n {N} -i 1000 -p 1"
    time = run(base.format(M=M, N=N))

    with open('results_cuda.txt', 'a') as f:
        w = csv.writer(f)
        w.writerow([(M, N), time])


