import numpy as np

import sys

import helper_functions as fcs

from codebase import metrics as mrx
seed = 420

random_state = np.random.RandomState(seed)

N= 25

#added_latents = range(int(N/3),int(2*N/3))

densities = np.arange(0.05,0.55,0.05)
density_ticks = densities #np.arange(0.1,1.1,0.1)

prob_bidirected = 0.9

corr_dict = {}

for p in densities:

    #experiment_list = []

    added_latents = range(int(p* N*(N-1) /6), int(N*(N-1) /3))

    number_of_experiments = 500

    #parent_sep = []

    #pparent_sep = []

    ZL_sep = []

    #MB_enhanced_parent_sep = []

    #MB_enhanced_pparent_sep = []

    #MB_enhanced_ZL_sep = []

    #parent_AID = []

    SHD = []

    for experiment in range(number_of_experiments):

        #start = time.time()
        G = fcs.generate_ER_random_mixed(N,p,prob_bidirected,random_state=random_state)
        H = fcs.generate_ER_random_mixed(N, p, prob_bidirected, random_state=random_state)
        #print('data generation')
        #print(time.time()-start)
        #start = time.time()
        ZL_sep.append(mrx.sym_SD_mixed_graphs(G,H,type='ZL',normalized = True, MB_enhanced = False))

        #MB_enhanced_parent_sep.append(mrx.sym_SD(G, H, type='parent', normalized=True, MB_enhanced=True))

        #parent_AID.append(mrx.sym_parent_AID_DAGs(G,H))
        #print('ZL')
        #print(time.time() - start)
        #start = time.time()
        SHD.append(mrx.SHD_MAGs(G,H))
        #print('SHD')
        #print(time.time() - start)
        #start = time.time()

    array = np.array([ZL_sep,SHD])


    corr_dict[p] = np.corrcoef(array)


corr_ZL_SD_SHD = []


for p in densities:

    corr_ZL_SD_SHD.append(corr_dict[p][0,1])



import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

ax1.plot(densities,corr_ZL_SD_SHD,label = 'ZL-SD and SHD')


ax1.set_xlabel('p', fontsize = 12)

ax1.set_ylabel('Correlation', fontsize = 12)
ax1.legend()

ax1.grid(True)

ax1.set_title('Correlation of Distances')


plt.xticks(density_ticks)

plt.yticks(np.arange(-0.4,1.0,0.1))

#plt.show()

filename = 'Plots/Correlations/'+ 'mixed_graphs_25_500_exps_symmetrized_q=0.9' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()