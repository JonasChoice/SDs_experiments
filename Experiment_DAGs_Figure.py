import numpy as np
import networkx as nx

import helper_functions as fcs
import copy

from codebase import metrics as mrx


seed = 420

random_state = np.random.RandomState(seed)

N= 25

densities = np.arange(0.05,1.0,0.05)
density_ticks = np.arange(0.0,1.1,0.1)


differences = {}
std = {}

for p in densities:

    experiment_list = []

    number_of_experiments = 100

    parent_sep = []

    pparent_sep = []

    ZL_sep = []

    MB_enhanced_parent_sep = []

    MB_enhanced_pparent_sep = []

    #MB_enhanced_ZL_sep = []

    parent_AID = []

    SHD_list = []

    parent_SD = []
    acyclic = 0
    loop_nb = 0

    while acyclic < number_of_experiments:
        loop_nb += 1
        G = fcs.generate_ER_random_DAG(N, p, random_state=random_state)
        H = copy.deepcopy(G)
        edges = list(H.directed.keys())
        if len(edges) == 0:
            continue
        ind = random_state.choice(len(edges))
        H.remove_directed(edges[ind][0], edges[ind][1])

        edges = list(H.directed.keys())
        ind = random_state.choice(len(edges))

        H.remove_directed(edges[ind][0], edges[ind][1])
        H.add_directed(edges[ind][1], edges[ind][0])

        if not nx.is_directed_acyclic_graph(H.to_nx()):
            continue

        acyclic += 1

        #CPDAG_G = G.get_CPDAG()
        #CPDAG_H = H.get_CPDAG()

        par_sep = mrx.SD_DAGs(G,H,type='parent',normalized = True, MB_enhanced = False)

        parent_sep.append(par_sep)

        ZL_sep.append(mrx.SD_DAGs(G,H,type='ZL',normalized = True, MB_enhanced = False))

        MB_enhanced_parent_sep.append(mrx.SD_DAGs(G, H, type='parent', normalized=True, MB_enhanced=True))

        par_AID = mrx.parent_AID_DAGs(G,H)

        parent_AID.append(par_AID)

        SHD_case = mrx.SHD_DAGs(G,H)
        #print(SHD_case)
        SHD_list.append(SHD_case)

        #parent_SD.append(abs(par_AID-SHD_case))

    #array = np.array([MB_enhanced_parent_sep,parent_AID,SHD])

        #if p == 1.0:
        #    print(par_sep,par_AID,SHD_case)

    print(p, acyclic, loop_nb)
    differences[p] = (np.mean(MB_enhanced_parent_sep),np.mean(parent_AID),np.mean(SHD_list),np.mean(parent_sep),np.mean(ZL_sep))
    std[p] = (np.std(MB_enhanced_parent_sep),np.std(parent_AID),np.std(SHD_list),np.std(parent_sep),np.std(ZL_sep))

MB_enhanced_SD = {}
MB_enhanced_SD_upper = {}
MB_enhanced_SD_lower = {}
AID = []
AID_upper = []
AID_lower = []
SHD = []
SHD_upper = []
SHD_lower = []
parent_SD_list = []
parent_SD_list_upper = []
parent_SD_list_lower = []

ZL_list = []
ZL_list_upper = []
ZL_list_lower = []

for p in densities:

    MB_enhanced_SD[p] = differences[p][0]
    MB_enhanced_SD_upper[p] = differences[p][0]+ std[p][0]
    MB_enhanced_SD_lower[p] = differences[p][0] - std[p][0]

    AID.append(differences[p][1])
    AID_upper.append(differences[p][1] + std[p][1])
    AID_lower.append(differences[p][1] - std[p][1])

    SHD.append(differences[p][2])
    SHD_upper.append(differences[p][2] + std[p][2])
    SHD_lower.append(differences[p][2] - std[p][2])

    parent_SD_list.append(differences[p][3])
    parent_SD_list_upper.append(differences[p][3]+ std[p][3])
    parent_SD_list_lower.append(differences[p][3] - std[p][3])

    ZL_list.append(differences[p][4])
    ZL_list_upper.append(differences[p][4] + std[p][4])
    ZL_list_lower.append(differences[p][4] - std[p][4])

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

ax1.plot(densities,parent_SD_list,color = '#CC79A7',label = 'pSD')
ax1.plot(densities,parent_SD_list_upper,color = '#CC79A7',alpha= 0.1)
ax1.plot(densities,parent_SD_list_lower,color = '#CC79A7',alpha= 0.1)
ax1.fill_between(densities,parent_SD_list_lower,parent_SD_list_upper,color = '#CC79A7',alpha= 0.1)

ax1.plot(densities,MB_enhanced_SD_upper.values(),color = 'lavender',alpha= 0.5)
ax1.plot(densities,MB_enhanced_SD_lower.values(),color = 'lavender',alpha= 0.5)
ax1.fill_between(densities,MB_enhanced_SD_lower.values(),MB_enhanced_SD_upper.values(),color = 'lavender',alpha= 0.5)
ax1.plot(densities,MB_enhanced_SD.values(),label = 'MB-pSD')

ax1.plot(densities,AID,label = 'pAID')
ax1.plot(densities,AID_upper,color = 'olive',alpha= 0.1)
ax1.plot(densities,AID_lower,color = 'olive',alpha= 0.1)
ax1.fill_between(densities,AID_lower,AID_upper,color = 'olive',alpha= 0.1)

ax1.plot(densities,SHD,label = 'SHD')
ax1.plot(densities,SHD_upper,color = '#D55E00',alpha= 0.1)
ax1.plot(densities,SHD_lower,color = '#D55E00',alpha= 0.1)
ax1.fill_between(densities,SHD_lower,SHD_upper,color = '#D55E00',alpha= 0.1)


ax1.plot(densities,ZL_list,color = 'y',label = 'ZL-SD')
ax1.plot(densities,ZL_list_upper,color = 'y',alpha= 0.1)
ax1.plot(densities,ZL_list_lower,color = 'y',alpha= 0.1)
ax1.fill_between(densities,ZL_list_lower,ZL_list_upper,color = 'y',alpha= 0.1)


ax1.set_xlabel('p', fontsize = 12)
ax1.grid(True)
ax1.set_ylabel('Average Distance', fontsize = 12)
ax1.legend()
#ax1.set_title('Average Differences of Symmetrized Distance Metrics')


plt.xticks(density_ticks)

plt.yticks(np.arange(0.0,0.4,0.1))

#plt.show()

filename = 'Plots/Differences/'+ 'experiments_DAGs_Figure_main_text_new' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()