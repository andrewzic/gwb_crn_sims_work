import numpy as np
import os
#import matplotlib
#matplotlib.use('Tkagg')
from matplotlib import pyplot as plt
from matplotlib import rc
from chainconsumer import ChainConsumer
from enterprise_warp import enterprise_warp
from enterprise_warp.results import parse_commandline
from enterprise_warp.results import EnterpriseWarpResult
from ptasim2enterprise import P02A
import json
# What results to grab
#psrs_set = '/home/bgonchar/correlated_noise_pta_2020/params/pulsar_set_x_1.dat'

#opts = parse_commandline()
#print(opts)
def interpret_opts_result(opts):
    """ Determine output directory from the --results argument
    from enterprise_warp.results script"""
    if os.path.isdir(opts.result):
      outdir_all = opts.result
    elif os.path.isfile(opts.result):
      params = enterprise_warp.Params(opts.result, init_pulsars=False)
      outdir_all = params.out + params.label_models + '_' + \
                        params.paramfile_label + '/'

      return outdir_all
    else:
      raise ValueError('--result seems to be neither a file, not a directory')


psrs_set = '/flush5/zic006/pptadr2_gwb_crn_sims/ptasim_26psr_similar/psrs.dat'
#result = ['/DATA/CETUS_3/zic006/ssb/ptasim/ptasim_26psr_similar/params/params_all_mc_array_spin_v_spincommon_20210524_r0.dat']
par = [
  ['gw'],
]
nmodel = [
1,
]
labels = [
"_", #"$n_\\mathrm{c} = 20$",
]
linestyles = [
"-",
]
shade = [
  False,
]
shade_alpha = [
0.0,
]
linewidths = [
0.5,
]
colors = [
"#1976D2",
]
extents=[[2.8, 5.2], [-15.35, -13.15]]

opts = parse_commandline()
opts.__dict__['logbf'] = True

#plt.style.use('seaborn-white')

fig, axes = plt.subplots(figsize = (3.31, 3.11))
cc = ChainConsumer()


#print(result)
#output_directory = interpret_opts_result(opts)

#for rr, pp, nm, ll in zip(result, par, nmodel, labels):
  #opts.__dict__['result'] = rr
  #opts.__dict__['par'] = pp
  #opts.__dict__['load_separated'] = 0
#  print(opts.__dict__)


if opts.par is None:
    opts.__dict__['par'] = 'gw'

print(opts.par)

result_obj = EnterpriseWarpResult(opts)
output_directory = result_obj.outdir_all
result_obj.main_pipeline()

nm = 1 #nmodel; CHANGE IF YOU WANT MODEL 0

if result_obj.counts is not None:
    print('HERE1')
    model_mask = np.round(result_obj.chain_burn[:,result_obj.ind_model]) == nm
    
    values = result_obj.chain_burn[model_mask,:]
    #values = values[10000:, :-4]
    values = values[:,result_obj.par_mask]
    print(values.shape)
else:
    values = result_obj.chain_burn[:,result_obj.par_mask]
    print(values.shape)
cc.add_chain(values, parameters=["$\gamma_{{\mathrm{{CP}}}}$", "$\log_{{10}} A_{{\mathrm{{CP}}}}$"])

cc.configure(summary=False, linestyles='-', linewidths=1.0,
             shade=True, bar_shade=True, shade_alpha=0.5, serif=True,
             legend_color_text=False, legend_artists=False, colors='#1976D2')
#cc.configure(summary=False, linestyles=linestyles, linewidths=linewidths, label_font_size = 12.0)#,
             #shade=shade, bar_shade=shade, shade_alpha=shade_alpha, serif=False,
             #legend_color_text=False, legend_artists=False, colors=colors)
             # legend_kwargs={"loc": "best"}, legend_location=(0, 0)
#rc("text", usetex=False)
#plt.rcParams['axes.grid'] = False
#cc.plotter.restore_rc_params()


#plt.vlines(np.log10(1.9e-15), ymin, ymax, linestyles="dotted", colors="black", label="NANOGrav 12.5-yr CPL")
#plt.ylabel('$\log_{{10}} A_{{\mathrm{{CP,m}}}}$')
#plt.xlabel('$\gamma_{{\mathrm{{CP,m}}}}$')

input_noise_pars = np.loadtxt('/flush5/zic006/pptadr2_gwb_crn_sims/ptasim_26psr_similar/psr_noise_vals/psr_noise_vals_1.dat')
input_gamms = input_noise_pars[:, 0]
input_as = input_noise_pars[:, 1]
input_wns = input_noise_pars[:, 2]

input_log10As = P02A(input_as, 0.01, -input_gamms)
#psrs = ["J0437-4715", "J1600-3053", "J1022+1001", "J1713+0747", "J1744-1134", "J1909-3744", "J2145-0750", "J2241-5236"]
#for p, a, g in zip(psrs, input_log10As, input_gamms):
#  print(p,a,g)
print(input_log10As)


marker_arr = np.array([input_gamms, input_log10As])
print(marker_arr.shape)
cfig = cc.plotter.plot(extents=extents, filename=output_directory+'log10_A_gamma.pdf')

for input_gamm, input_log10A in zip(input_gamms, input_log10As):
  cc.add_marker([input_gamm, input_log10A], parameters=["$\gamma_{{\mathrm{{CP}}}}$", "$\log_{{10}} A_{{\mathrm{{CP}}}}$"] )


#, truth={"gamma": 13./3.})
#cc.configure(summary=False, linestyles=linestyles, linewidths=linewidths)#,  
fig = plt.gcf()
fig.set_size_inches(4,4)
axes = fig.axes
# cs = ['C0', 'C1', 'C2', 'C3']
print(axes)
axes[2].clear()




psr_noise_dict = json.load(open('{}/noisefiles/_noise.json'.format(output_directory), 'r'))

rednoise_log10_As = []
rednoise_gammas = []
for key, item in psr_noise_dict.items():
    if "red_noise" in key:
        if "gamma" in key:
            rednoise_gammas.append(item)
        if "log10_A" in key:
            rednoise_log10_As.append(item)

gw_gamma = psr_noise_dict["gw_gamma"]
gw_log10_A = psr_noise_dict["gw_log10_A"]

cc.plotter.plot_contour(ax = axes[2], parameter_x = "$\gamma_{{\mathrm{{CP}}}}$", parameter_y = "$\log_{{10}} A_{{\mathrm{{CP}}}}$")

print(gw_gamma)

axes[2].plot(-input_gamms, input_log10As, color = 'C1', marker = 'X', zorder = 10000, linestyle = '')

axes[2].set_xlabel("$\gamma_{{\mathrm{{CP}}}}$", fontsize = 12)
axes[2].set_ylabel("$\log_{{10}} A_{{\mathrm{{CP}}}}$", fontsize = 12)
axes[2].set_xlim(2.8, 5.2)#np.min(gw_gamma) - 1.1, np.max(gw_gamma) + 1.1)
axes[2].set_ylim(-15.35, -13.15)
#print(axes)


axes[0].set_xlim(axes[2].get_xlim())
axes[3].set_ylim(axes[2].get_ylim())
for input_gamm in input_gamms:
  axes[0].axvline(-input_gamm, c = 'C1', alpha = 0.4, ls = '--')

for input_logA in input_log10As:
  axes[3].axhline(input_logA, c = 'C1', alpha = 0.4, ls = '--')

plt.rcParams['figure.constrained_layout.use'] = True
#axes[2].set_xlim(3, 5)#np.min(gw_gamma) - 1.1, np.max(gw_gamma) + 1.1)
#axes[2].set_ylim(-15.25, -14.25)
print('old ylims', 1.05*np.min(input_log10As), 1.05*np.max(input_log10As))
#axes[2].set_ylim(1.05*np.min(input_log10As), 1.05*np.max(input_log10As))
#axes[2].set_ylim(np.min(input_log10As) - .8, np.max(input_log10As) + .8)


#plt.ylabel('Posterior probability')
#plt.xlim([-17,-12])
#plt.ylim([ymin, ymax])
#plt.yscale("log")
#plt.legend()
#plt.grid(b=None)
#plt.tight_layout()
#fig.tight_layout()
#extents = 
cfig = cc.plotter.plot(extents=extents, filename=output_directory+'log10_A_gamma.pdf')
fig = plt.gcf()
fig.set_size_inches(3.31, 3.11)
fig.savefig(output_directory + 'log10_A_gamma2.png', bbox_inches = 'tight', dpi = 300)
fig.savefig(output_directory + 'log10_A_gamma2.pdf', bbox_inches = 'tight')
#plt.show()
