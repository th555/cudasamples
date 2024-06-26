import matplotlib.pyplot as plt

res = [((100, 100),(0.235612, 509354400.0)),
((1000, 1000),(0.33553, 35767290000.0)),
((100, 20000),(0.456647, 52561390000.0)),
((20000, 100),(0.418624, 57335460000.0)),
((5000, 5000),(2.702936, 110999700000.0))]

dims, ress = zip(*res)
time, flops = zip(*ress)
flops = [f/1e9 for f in flops]

plt.bar([str(k) for k in dims], time)
plt.xlabel('cylinder dimensions (M, N)')
plt.ylabel('runtime (s)')
plt.title('runtime of CUDA heat dissipation program')
plt.show()
