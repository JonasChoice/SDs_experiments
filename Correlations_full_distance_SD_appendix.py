import numpy as np

import sys

import helper_functions as fcs

from codebase import metrics as mrx
seed = 42

random_state = np.random.RandomState(seed)

N= 10

densities = np.arange(0.05,0.55,0.05)
density_ticks = densities  #np.arange(0.1,1.0,0.1)


corr_dict = {}

dist_dict = {}

for p in densities:

    experiment_list = []

    number_of_experiments = 100

    parent_sep = []

    #pparent_sep = []

    ZL_sep = []

    MB_enhanced_parent_sep = []

    #MB_enhanced_pparent_sep = []

    #MB_enhanced_ZL_sep = []

    #parent_AID = []

    #SHD = []

    full_distance = []

    for experiment in range(number_of_experiments):

        G = fcs.generate_ER_random_DAG(N,p,random_state=random_state)
        H = fcs.generate_ER_random_DAG(N, p, random_state=random_state)

        full_distance.append(mrx.metric_DAGs(H,G,type = 's', max_order=None, randomize_higher_order = 0,normalized= True, random_state = None, include_distance_dict = False))


        parent_sep.append(mrx.SD_DAGs(G,H,type='parent',normalized = True, MB_enhanced = False))

        MB_enhanced_parent_sep.append(mrx.SD_DAGs(G, H, type='parent', normalized=True, MB_enhanced=True))

        ZL_sep.append(mrx.SD_DAGs(G,H,type='ZL',normalized = True, MB_enhanced = False))


    array = np.array([full_distance,parent_sep, MB_enhanced_parent_sep, ZL_sep])




    corr_dict[p] = np.corrcoef(array)

corr_parent_SD = []

corr_MB_enhanced = []
corr_ZL = []

for p in densities:

    corr_parent_SD.append(corr_dict[p][0, 1])

    corr_MB_enhanced.append(corr_dict[p][0, 2])

    corr_ZL.append(corr_dict[p][0,3])


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

ax1.plot(densities,corr_parent_SD,label = 'pSD')
ax1.plot(densities,corr_MB_enhanced,label = 'MB-pSD')
ax1.plot(densities,corr_ZL,label = 'ZL_SD')
#ax1.plot(densities,corr_parent_AID_SHD,label = 'pAID and SHD')


ax1.set_xlabel('p', fontsize = 12)
ax1.grid(True)
ax1.set_ylabel('Correlation', fontsize = 12)
ax1.legend()
#ax1.set_title('Correlation with full s-distance')


plt.xticks(density_ticks)

plt.yticks(np.arange(-0.8,1.2,0.2))

#plt.show()

filename = 'Plots/Correlations/'+ 'correlation_with_full_s-distance' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()