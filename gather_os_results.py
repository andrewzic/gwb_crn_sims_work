import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import ptasim2enterprise as p2e

import argparse
import glob


realisations = np.arange(0, 10, 1)

psr_list = np.loadtxt('psrs.dat', dtype = 'str')

N_psr = len(psr_list)

np.random.seed(seed = 20210524)

ptasim_inp_template_fn = 'ptasim_input_files/ptasim_all_similar_26_N_template.inp'
ptasim_inp_template_f = open(ptasim_inp_template_fn, 'r')
ptasim_inp_template_str = ptasim_inp_template_f.read()

#print(ptasim_inp_template.format('test', 'ts', '2ff'))

alpha_uppers = np.linspace(0, -4, 11)[::-1]
alpha_lowers = np.linspace(-8, -4, 11)[::-1]

p0_lowers = np.linspace(-30, -23, 11)[::-1]
p0_uppers = np.linspace(-16, -23, 11) [::-1]

Alpha_lowers, P0_lowers = np.meshgrid(alpha_lowers, p0_lowers)
Alpha_uppers, P0_uppers = np.meshgrid(alpha_uppers, p0_uppers)

N = np.arange(0, p0_lowers.shape[0])

dalphas = []
dP0s = []
dlog10_As = []
for y in N:

    alphas = np.random.uniform(alpha_lowers[y], alpha_uppers[y], size = N_psr)
    dalpha = alpha_uppers[y] - alpha_lowers[y]
    #toaerrs = np.random.uniform(9e-8, 5e-7, size = N_psr)
    dalphas.append(dalpha)

dalphas = np.array(sorted(dalphas))
for x in N:

    p0s = 10**(np.random.uniform(p0_lowers[x], p0_uppers[x], size = N_psr))
    log10As = p2e.P02A(p0s, 0.01, -alphas)
    print(alpha_lowers[x])
    log10A_lower = p2e.P02A(10**p0_lowers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    log10A_upper = p2e.P02A(10**p0_uppers[x], 0.01, -1.0*np.array([alpha_lowers[x], alpha_uppers[x]]))
    if x == 0:
        print('HFWBFKNJKFAD', log10A_lower, log10A_upper)
    dP0 = p0_uppers[x] - p0_lowers[x]
    dP0s.append(dP0)
    dlog10_A = np.max(log10A_upper) - np.min(log10A_lower)
    dlog10_As.append(dlog10_A)

dP0s = np.array(sorted(dP0s))
dlog10_As = np.array(dlog10_As)

os_matrix_list = []
os_marg_matrix_list = []
os_snr_matrix_list = []
os_marg_snr_matrix_list = []

m_os_matrix =          np.nan*np.zeros_like(Alpha_lowers)
m_os_marg_matrix =     np.nan*np.zeros_like(Alpha_lowers)
m_os_snr_matrix =      np.nan*np.zeros_like(Alpha_lowers)
m_os_marg_snr_matrix = np.nan*np.zeros_like(Alpha_lowers)


for realisation_ind in realisations:


  result_dirs = sorted(glob.glob('enterprise_out/mc_array_spin_v_spincommon_*r{}/*/'.format(realisation_ind))) #parent dirs
  result_dP0s = [float(i.split('/')[-2].split('_')[-3]) for i in result_dirs]
  result_dalphas = [float(i.split('/')[-2].split('_')[-2]) for i in result_dirs]

  os_pickle_files = [i + '_os_results.pkl' for i in result_dirs]
  chain_files = [i + '/chain_1.txt' for i in result_dirs]
  for os_pickle_file, result_dP0, result_dalpha, chain in zip(os_pickle_files, result_dP0s, result_dalphas, chain_files):

    os_matrix = np.nan*np.zeros_like(Alpha_lowers)
    os_marg_matrix = np.nan*np.zeros_like(Alpha_lowers)
    os_snr_matrix = np.nan*np.zeros_like(Alpha_lowers)
    os_marg_snr_matrix = np.nan*np.zeros_like(Alpha_lowers)

    dP0_ind = np.argmin(np.abs(dP0s - result_dP0))
    dalpha_ind = np.argmin(np.abs(dalphas - result_dalpha))
    chainsize =os.path.getsize(chain)
    if chainsize < 1E8:
      continue
    with open(os_pickle_file, 'rb') as os_pickle:
    
      os_result = pickle.load(os_pickle)
      print(os_pickle_file)
      os_result = os_result['monopole']
      os_OS = os_result['OS']
      os_OS_err = os_result['OS_err']
      os_SNR = os_OS/os_OS_err
      os_marg_OS = os_result['marginalised_os']
      os_marg_OS_err = os_result['marginalised_os_err']
      mean_marg_OS = np.nanmean(os_marg_OS)
      mean_marg_SNR = np.nanmean(os_marg_OS/os_marg_OS_err)

      os_matrix[dP0_ind, dalpha_ind] = os_OS
      os_marg_matrix[dP0_ind, dalpha_ind] = mean_marg_OS
      os_snr_matrix[dP0_ind, dalpha_ind] = os_SNR
      os_marg_snr_matrix[dP0_ind, dalpha_ind] = mean_marg_SNR
    os_matrix_list.append(os_matrix)
    os_marg_matrix_list.append(os_matrix_list)
    os_snr_matrix_list.append(os_snr_matrix)
    os_marg_snr_matrix_list.append(os_marg_snr_matrix)
    print('doing dstacks')
    m_os_matrix =          np.dstack([m_os_matrix, os_matrix])
    m_os_marg_matrix =     np.dstack([m_os_marg_matrix, os_marg_matrix])
    m_os_snr_matrix =      np.dstack([m_os_snr_matrix, os_snr_matrix])
    m_os_marg_snr_matrix = np.dstack([m_os_marg_snr_matrix, os_marg_snr_matrix])

m_os_matrix =          np.nanmean(m_os_matrix[:, :, 1:], axis = 2) #np.array(os_matrix_list)[:, :, 1:], axis = 0)
m_os_marg_matrix =     np.nanmean(m_os_marg_matrix[:, :, 1:], axis = 2)#np.array(os_marg_matrix_list), axis = 0)
m_os_snr_matrix =      np.nanmean(m_os_snr_matrix[:, :, 1:], axis = 2)#np.array(os_snr_matrix_list), axis = 0)
m_os_marg_snr_matrix = np.nanmean(m_os_marg_snr_matrix[:, :, 1:], axis = 2)#np.array(os_marg_snr_matrix_list), axis = 0)

import matplotlib.colors
from matplotlib.ticker import LogLocator

plt.rcParams.update({
  "text.usetex": True,
  "font.family": "serif",
})

font = {'family' : 'serif',
        'size'   : 17}

fig, ax = plt.subplots(1,1)
im = plt.imshow(np.abs(m_os_matrix), cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dP0s[[0,-1]]], aspect = 'auto', interpolation = 'none')#, clim = [1E-6, 1E6]) #
#im = plt.imshow(bf_matrix, cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
cb.set_label('$\hat{{A_{\mathrm{mp}}}}^{{2}}$', fontsize = font['size'])
cb.ax.minorticks_on()
#minorticks = im.norm(np.arange(1E-6, 1E6, 1))
#cb.ax.yaxis.set_ticks(minorticks, minor = True)
ax.set_xlabel(r'$\Delta \alpha$', fontdict = font)
ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.minorticks_on()
plt.savefig('os_matrix_mp.png', dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(1,1)
im = plt.imshow(np.abs(m_os_marg_matrix), cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dP0s[[0,-1]]], aspect = 'auto', interpolation = 'none')#, clim = [1E-6, 1E6]) #
#im = plt.imshow(bf_matrix, cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
cb.set_label('$\hat{{A_{\mathrm{mp}}}}^{{2}}$', fontsize = font['size'])
cb.ax.minorticks_on()
#minorticks = im.norm(np.arange(1E-6, 1E6, 1))
#cb.ax.yaxis.set_ticks(minorticks, minor = True)
ax.set_xlabel(r'$\Delta \alpha$', fontdict = font)
ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.minorticks_on()
plt.savefig('os_marg_matrix_mp.png', dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(1,1)
im = plt.imshow(np.abs(m_os_snr_matrix), cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dP0s[[0,-1]]], aspect = 'auto', interpolation = 'none')#, clim = [1E-6, 1E6]) #
#im = plt.imshow(bf_matrix, cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
cb.set_label('S/N(mp)', fontsize = font['size'])
cb.ax.minorticks_on()
#minorticks = im.norm(np.arange(1E-6, 1E6, 1))
#cb.ax.yaxis.set_ticks(minorticks, minor = True)
ax.set_xlabel(r'$\Delta \alpha$', fontdict = font)
ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.minorticks_on()
plt.savefig('os_snr_matrix_mp.png', dpi = 300, bbox_inches = 'tight')

fig, ax = plt.subplots(1,1)
im = plt.imshow(np.abs(m_os_marg_snr_matrix), cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dP0s[[0,-1]]], aspect = 'auto', interpolation = 'none')#, clim = [1E-6, 1E6]) #
#im = plt.imshow(bf_matrix, cmap = 'inferno', norm = matplotlib.colors.LogNorm(), origin = 'lower', extent = [*dalphas[[0,-1]], *dlog10_As[[0,-1]]], aspect = 'auto', clim = [1E-6, 1E6], interpolation = 'none') #

cb = plt.colorbar(im)#, ticks = LogLocator(subs=range(10)))
cb.set_label('S/N(mp)', fontsize = font['size'])
cb.ax.minorticks_on()
#minorticks = im.norm(np.arange(1E-6, 1E6, 1))
#cb.ax.yaxis.set_ticks(minorticks, minor = True)
ax.set_xlabel(r'$\Delta \alpha$', fontdict = font)
ax.set_ylabel(r'$\Delta \log_{{10}}(P_0)$', fontdict = font)
ax.tick_params(axis='y', labelsize = font['size'])
ax.tick_params(axis='x', labelsize = font['size'])
plt.minorticks_on()
plt.savefig('os_marg_snr_matrix_mp.png', dpi = 300, bbox_inches = 'tight')



# plt.savefig('bf_matrix.pdf', bbox_inches = 'tight')
# plt.savefig('bf_matrix.png', bbox_inches = 'tight', dpi = 300)
