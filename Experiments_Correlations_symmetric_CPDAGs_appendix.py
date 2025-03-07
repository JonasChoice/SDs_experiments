import numpy as np

import sys

import helper_functions as fcs

from codebase import metrics as mrx
seed = 4422

random_state = np.random.RandomState(seed)

N= 25

densities = np.arange(0.05,1.0,0.05)
density_ticks = np.arange(0.1,1.1,0.1)


corr_dict = {}

for p in densities:

    experiment_list = []

    number_of_experiments = 500

    parent_sep = []

    pparent_sep = []

    #ZL_sep = []

    MB_enhanced_parent_sep = []

    MB_enhanced_pparent_sep = []

    #MB_enhanced_ZL_sep = []

    parent_AID = []

    SHD = []

    for experiment in range(number_of_experiments):
        #start = time.time()
        G = fcs.generate_ER_random_DAG(N,p,random_state=random_state).get_CPDAG()
        H = fcs.generate_ER_random_DAG(N, p, random_state=random_state).get_CPDAG()
        #print('data generation')
        #print(p, time.time() - start)
        #start = time.time()
        parent_sep.append(mrx.sym_SD_CPDAGs(G,H,type='pparent',normalized = True, MB_enhanced = False))
        #print('pparent')
        #print(time.time() - start)
        #start = time.time()
        MB_enhanced_parent_sep.append(mrx.sym_SD_CPDAGs(G, H, type='pparent', normalized=True, MB_enhanced=True))
        #print('MB_enhanced')
        #print(time.time() - start)
        #start = time.time()
        parent_AID.append(mrx.sym_parent_AID_CPDAGs(G,H))

        SHD.append(mrx.SHD_CPDAGs(G,H))
        #print('AID SHD')
        #print(time.time() - start)

    array = np.array([parent_sep,MB_enhanced_parent_sep,parent_AID,SHD])


    corr_dict[p] = np.corrcoef(array)

corr_parent_SD_MB_enhanced_SD = []
corr_parent_SD_AID = []
corr_parent_SD_SHD = []
corr_parent_AID_SHD = []

for p in densities:

    corr_parent_SD_MB_enhanced_SD.append(corr_dict[p][0, 1])

    corr_parent_SD_AID.append(corr_dict[p][0, 2])

    corr_parent_SD_SHD.append(corr_dict[p][0,3])

    corr_parent_AID_SHD.append(corr_dict[p][2, 3])

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

ax1.plot(densities,corr_parent_SD_MB_enhanced_SD,label = 'ppSD and MB-ppSD')
ax1.plot(densities,corr_parent_SD_AID,label = 'ppSD and pAID')
ax1.plot(densities,corr_parent_SD_SHD,label = 'ppSD and SHD')
ax1.plot(densities,corr_parent_AID_SHD,label = 'pAID and SHD')


ax1.set_xlabel('p', fontsize = 12)

ax1.set_ylabel('Correlation', fontsize = 12)
ax1.grid(True)
ax1.legend()
ax1.set_title('Correlation of Symmetrized Distances')


plt.xticks(density_ticks)
plt.yticks(np.arange(-0.8,1.2,0.2))
#plt.show()

filename = 'Plots/Correlations/'+ 'symmetrized_CPDAGs_25_500_exps' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()