import numpy as np
import matplotlib.pyplot as plt

psr_list = np.loadtxt('psrs.dat', dtype = 'str')

N_psr = len(psr_list)

np.random.seed(seed = 20210524)

alphas = np.random.uniform(-5, -3, size = N_psr)
p0s = 10**(np.random.uniform(-24, -22.2, size = N_psr))
toaerrs = np.random.uniform(9e-8, 5e-7, size = N_psr)

fmt_str_tnoise = "tnoise: psr={:s},alpha={:.1f},p0={:.1e},fc=0.01"
fmt_str_obs = "observe: psr={:s},toaerr={:.1e},freq=1400"
fmt_str_dat = "{:.4f}\t{:.4e}\t{:.4e}\n"


for psr, alpha, p0 in zip(psr_list, alphas, p0s):
    print(fmt_str_tnoise.format(psr, alpha, p0))

for psr, toaerr in zip(psr_list, toaerrs):
    print(fmt_str_obs.format(psr, toaerr))

with open('psr_noise_vals/psr_noise_vals.dat', 'w') as psr_datf:
    for alpha, p0, toaerr in zip(alphas, p0s, toaerrs):
        psr_datf.write(fmt_str_dat.format(alpha, p0, toaerr))

    psr_datf.close()

plt.scatter(alphas, p0s)

plt.show()
