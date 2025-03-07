
import numpy as np

import sys

import helper_functions as fcs

from codebase import metrics as mrx


seed = 42

random_state = np.random.RandomState(seed)

N= 25

densities = np.arange(0.05,1.0,0.05)

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

        G = fcs.generate_ER_random_DAG(N,p,random_state=random_state)
        H = fcs.generate_ER_random_DAG(N, p, random_state=random_state)

        parent_sep.append(mrx.SD_DAGs(G,H,type='parent',normalized = True, MB_enhanced = False))

        MB_enhanced_parent_sep.append(mrx.SD_DAGs(G, H, type='parent', normalized=True, MB_enhanced=True))

        parent_AID.append(mrx.parent_AID_DAGs(G,H))

        SHD.append(mrx.SHD_DAGs(G,H))

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

ax1.plot(densities,corr_parent_SD_MB_enhanced_SD,label = 'pSD and MB-pSD')
ax1.plot(densities,corr_parent_SD_AID,label = 'pSD and pAID')
ax1.plot(densities,corr_parent_SD_SHD,label = 'pSD and SHD')
ax1.plot(densities,corr_parent_AID_SHD,label = 'pAID and SHD')


ax1.set_xlabel('p', fontsize = 12)
ax1.grid(True)
ax1.set_ylabel('Correlation', fontsize = 12)
ax1.legend()
ax1.set_title('Correlation of Distances')


plt.xticks(np.arange(0.0,1.1,0.1))

#plt.show()

filename = 'Plots/Correlations/'+ 'DAGs_size_25_500_exps' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()