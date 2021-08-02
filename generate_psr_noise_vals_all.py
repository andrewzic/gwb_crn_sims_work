import numpy as np
import matplotlib.pyplot as plt
import os
import ptasim2enterprise as p2e

psr_list = np.loadtxt('psrs.dat', dtype = 'str')

N_psr = len(psr_list)

np.random.seed(seed = 20210524)

ptasim_inp_template_fn = 'ptasim_input_files/ptasim_all_similar_26_N_template.inp'
ptasim_inp_template_f = open(ptasim_inp_template_fn, 'r')
ptasim_inp_template_str = ptasim_inp_template_f.read()

#print(ptasim_inp_template.format('test', 'ts', '2ff'))

N = [0, 1, 10, 11, 2, 3]

alpha_uppers = [-4, -3, -2.8, -2.4, -2, -1]
alpha_lowers = [-4, -5, -5.2, -5.6, -6, -7]

p0_uppers = [-23, -22, -21.8, -21.5, -21, -20]
p0_lowers = [-23, -24, -24.2, -24.5, -25, -26]

for ind, alpha_upper, alpha_lower, p0_upper, p0_lower in zip(N, alpha_uppers, alpha_lowers, p0_uppers, p0_lowers):
    alphas = np.random.uniform(alpha_lower, alpha_upper, size = N_psr)
    p0s = 10**(np.random.uniform(p0_lower, p0_upper, size = N_psr))
    toaerrs = np.random.uniform(9e-8, 5e-7, size = N_psr)
    log10As = p2e.P02A(p0s, 0.01, -alphas)
    log10A_lower = p2e.P02A(10**p0_lower, 0.01, -1.0*np.array([alpha_lower, alpha_upper]))
    log10A_upper = p2e.P02A(10**p0_upper, 0.01, -1.0*np.array([alpha_lower, alpha_upper]))
    print(ind)
    print(log10A_lower)
    print(log10A_upper)
    #print(log10As)

    fmt_str_tnoise = "tnoise: psr={:s},alpha={:.1f},p0={:.1e},fc=0.01\n"
    fmt_str_obs = "observe: psr={:s},toaerr={:.1e},freq=1400\n"
    fmt_str_dat = "{:.4f}\t{:.4e}\t{:.4e}\n"
    
    str_tnoise = ''
    for psr, alpha, p0 in zip(psr_list, alphas, p0s):
        str_tnoise += fmt_str_tnoise.format(psr, alpha, p0)
        #print(fmt_str_tnoise.format(psr, alpha, p0))
        
    str_obs = ''
    for psr, toaerr in zip(psr_list, toaerrs):
        str_obs += fmt_str_obs.format(psr, toaerr)
        #print(fmt_str_obs.format(psr, toaerr))

    
    if not os.path.exists('ptasim_input_files/ptasim_all_similar_26_{}.inp'.format(ind)):
        with open('ptasim_input_files/ptasim_all_similar_26_{}.inp'.format(ind), 'w') as ptasim_inp_f:
            ptasim_inp_f.write(ptasim_inp_template_str.format(ind, str_tnoise, str_obs))
            ptasim_inp_f.close()
        
        with open('psr_noise_vals/psr_noise_vals_{}.dat'.format(ind), 'w') as psr_datf:
            for alpha, p0, toaerr in zip(alphas, p0s, toaerrs):
                psr_datf.write(fmt_str_dat.format(alpha, p0, toaerr))

            psr_datf.close()

#plt.scatter(alphas, p0s)

#plt.show()
