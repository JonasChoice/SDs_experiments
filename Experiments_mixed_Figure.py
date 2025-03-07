
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
prob_bidirected = 0.25

differences = {}
std = {}

for p in densities:

    experiment_list = []

    number_of_experiments = 100

    #parent_sep = []

    #pparent_sep = []

    ZL_sep = []

    #MB_enhanced_parent_sep = []

    #MB_enhanced_pparent_sep = []

    MB_enhanced_ZL_sep = []

    #parent_AID = []

    SHD_list = []

    #parent_SD = []

    acyclic = 0
    loop_nb = 0

    while acyclic < number_of_experiments:
        loop_nb += 1

        G = fcs.generate_ER_random_mixed(N,p,prob_bidirected,random_state = random_state)
        H = copy.deepcopy(G)
        edges = list(H.directed.keys())
        if len(edges) == 0:
            continue
        ind = random_state.choice(len(edges))
        H.remove_directed(edges[ind][0], edges[ind][1])

        edges = list(H.directed.keys())
        if len(edges) == 0:
            continue
        ind = random_state.choice(len(edges))

        H.remove_directed(edges[ind][0], edges[ind][1])
        H.add_directed(edges[ind][1], edges[ind][0])

        H_help = copy.deepcopy(H)

        H_help.remove_all_bidirected()

        if not nx.is_directed_acyclic_graph(H_help.to_nx()):
            continue

        acyclic += 1

        bi_edges = list(H.bidirected.keys())
        if len(bi_edges) == 0:
            continue
        ind = random_state.choice(len(bi_edges))
        bi_edge = list(bi_edges[ind])
        H.remove_bidirected(bi_edge[0], bi_edge[1])

        #print(H)

        ZL_sep.append(mrx.SD_mixed_graphs(G,H,type='ZL',normalized = True, MB_enhanced = False))



        #print('MAG', mrx.SD_mixed_graphs(G,H,type='ZL',normalized = True, MB_enhanced = False))
        #print('DAG', mrx.SD_DAGs(G,H,type='ZL',normalized = True, MB_enhanced = False))


        SHD_case = mrx.SHD_MAGs(G,H)
        SHD_list.append(SHD_case)

    print(p, acyclic, loop_nb)
    differences[p] = (np.mean(ZL_sep),np.mean(SHD_list))
    std[p] = (np.std(ZL_sep),np.std(SHD_list))



#MB_enhanced_SD = {}
#MB_enhanced_SD_upper = {}
#MB_enhanced_SD_lower = {}
SHD = []
SHD_upper = []
SHD_lower = []

ZL_list = []
ZL_list_upper = []
ZL_list_lower = []

for p in densities:
    ZL_list.append(differences[p][0])
    ZL_list_upper.append(differences[p][0] + std[p][0])
    ZL_list_lower.append(differences[p][0] - std[p][0])

    SHD.append(differences[p][1])
    SHD_upper.append(differences[p][1] + std[p][1])
    SHD_lower.append(differences[p][1] - std[p][1])


import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)



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

filename = 'Plots/Differences/'+ 'experiments_mixed_Figure_main_text' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()