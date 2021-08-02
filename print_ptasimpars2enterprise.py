import numpy as np
from ptasim2enterprise import P02A

fc = 0.01

arr = np.loadtxt('similar_params3_smp.dat')

alpha = arr[:, 0]
p0 = arr[:,1]

arr_ent = arr.copy()

ent_A = P02A(p0, fc, alpha)

arr_ent[:,1] = ent_A

for row in arr_ent:
    print('{} {}'.format(row[0], row[1]))
