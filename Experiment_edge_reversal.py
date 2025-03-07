import numpy as np
import networkx as nx
import sys
import helper_functions as fcs
import copy

from codebase import metrics as mrx


seed = 4243

random_state = np.random.RandomState(seed)

N= 25
p=0.25

parent_sep = {}

ZL_sep = {}

MB_enhanced_parent_sep = {}

parent_AID = {}

SHD_list = {}

experiment_dict = {}

number_of_experiments = 100

number_of_reversals = 10

all_acyclic = 0
loop_nb = 0
attempts =  100

while all_acyclic < number_of_experiments:

    graphs = []
    graphs.append(fcs.generate_ER_random_DAG(N, p, random_state=random_state))
    for r in range(number_of_reversals):
        t = 0
        while t < attempts:
            try_graph = copy.deepcopy(graphs[r])

            edges = list(try_graph.directed.keys())
            if len(edges) == 0:
                continue
            ind = random_state.choice(len(edges))
            try_graph.remove_directed(edges[ind][0], edges[ind][1])


            edges = list(try_graph.directed.keys())
            ind = random_state.choice(len(edges))



            try_graph.remove_directed(edges[ind][0], edges[ind][1])
            try_graph.add_directed(edges[ind][1], edges[ind][0])
            t += 1
            if nx.is_directed_acyclic_graph(try_graph.to_nx()):
                graphs.append(try_graph)
                break

            #print(r,t)


    #print(graphs[-1])
    all_acyclic += 1


    parent_sep[all_acyclic] = []
    ZL_sep[all_acyclic] = []
    MB_enhanced_parent_sep[all_acyclic] = []
    parent_AID[all_acyclic] = []
    SHD_list[all_acyclic] = []


    for r in range(1,number_of_reversals+1):
        parent_sep[all_acyclic].append(mrx.SD_DAGs(graphs[0],graphs[r],type='parent',normalized = True, MB_enhanced = False))
        ZL_sep[all_acyclic].append(mrx.SD_DAGs(graphs[0],graphs[r], type='ZL', normalized=True, MB_enhanced=False))

        MB_enhanced_parent_sep[all_acyclic].append(mrx.SD_DAGs(graphs[0],graphs[r], type='parent', normalized=True, MB_enhanced=True))

        parent_AID[all_acyclic].append(mrx.parent_AID_DAGs(graphs[0],graphs[r]))

        SHD_list[all_acyclic].append(mrx.SHD_DAGs(graphs[0],graphs[r]))





parent_sep_mean = {}

ZL_sep_mean = {}

MB_enhanced_parent_sep_mean = {}

parent_AID_mean = {}

SHD_mean = {}

parent_sep_upper = {}

ZL_sep_upper = {}

MB_enhanced_parent_sep_upper = {}

parent_AID_upper = {}

SHD_upper = {}

parent_sep_lower = {}

ZL_sep_lower = {}

MB_enhanced_parent_sep_lower = {}

parent_AID_lower = {}

SHD_lower = {}

for r in range(1,number_of_reversals+1):

    parent_sep_list = []

    ZL_sep_list = []

    MB_enhanced_parent_sep_list = []

    parent_AID_list = []

    SHD_list_2 = []

    for exp in range(1,number_of_experiments+1):
        parent_sep_list.append(parent_sep[exp][r-1])
        ZL_sep_list.append(ZL_sep[exp][r-1])
        MB_enhanced_parent_sep_list.append(MB_enhanced_parent_sep[exp][r-1])
        parent_AID_list.append(parent_AID[exp][r-1])
        SHD_list_2.append(SHD_list[exp][r-1])

    parent_sep_mean[r] = np.mean(parent_sep_list)
    ZL_sep_mean[r] = np.mean(ZL_sep_list)
    MB_enhanced_parent_sep_mean[r] = np.mean(MB_enhanced_parent_sep_list)
    parent_AID_mean[r] = np.mean(parent_AID_list)
    SHD_mean[r] = np.mean(SHD_list_2)

    parent_sep_upper[r] = parent_sep_mean[r] + np.std(parent_sep_list)
    ZL_sep_upper[r] = ZL_sep_mean[r] + np.std(ZL_sep_list)
    MB_enhanced_parent_sep_upper[r] =  MB_enhanced_parent_sep_mean[r] + np.std(MB_enhanced_parent_sep_list)
    parent_AID_upper[r] = parent_AID_mean[r] +  np.std(parent_AID_list)
    SHD_upper[r] =  SHD_mean[r] + np.std(SHD_list_2)

    parent_sep_lower[r] = parent_sep_mean[r] - np.std(parent_sep_list)
    ZL_sep_lower[r] = ZL_sep_mean[r] - np.std(ZL_sep_list)
    MB_enhanced_parent_sep_lower[r] = MB_enhanced_parent_sep_mean[r] - np.std(MB_enhanced_parent_sep_list)
    parent_AID_lower[r] = parent_AID_mean[r] - np.std(parent_AID_list)
    SHD_lower[r] = SHD_mean[r] - np.std(SHD_list_2)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ax1 = plt.subplots(1,1,sharex=True, sharey=True)

densities = range(1,number_of_reversals+1)
print(densities,parent_sep_mean)
ax1.plot(densities,parent_sep_mean.values(),color = '#CC79A7',label = 'pSD')
ax1.plot(densities,parent_sep_upper.values(),color = '#CC79A7',alpha= 0.1)
ax1.plot(densities,parent_sep_lower.values(),color = '#CC79A7',alpha= 0.1)
ax1.fill_between(densities,parent_sep_lower.values(),parent_sep_upper.values(),color = '#CC79A7',alpha= 0.1)

ax1.plot(densities,MB_enhanced_parent_sep_upper.values(),color = 'lavender',alpha= 0.5)
ax1.plot(densities,MB_enhanced_parent_sep_lower.values(),color = 'lavender',alpha= 0.5)
ax1.fill_between(densities,MB_enhanced_parent_sep_lower.values(),MB_enhanced_parent_sep_upper.values(),color = 'lavender',alpha= 0.5)
ax1.plot(densities,MB_enhanced_parent_sep_mean.values(),label = 'MB-pSD')

ax1.plot(densities,parent_AID_mean.values(),label = 'pAID')
ax1.plot(densities,parent_AID_upper.values(),color = 'olive',alpha= 0.1)
ax1.plot(densities,parent_AID_lower.values(),color = 'olive',alpha= 0.1)
ax1.fill_between(densities,parent_AID_lower.values(),parent_AID_upper.values(),color = 'olive',alpha= 0.1)

ax1.plot(densities,SHD_mean.values(),label = 'SHD')
ax1.plot(densities,SHD_upper.values(),color = '#D55E00',alpha= 0.1)
ax1.plot(densities,SHD_lower.values(),color = '#D55E00',alpha= 0.1)
ax1.fill_between(densities,SHD_lower.values(),SHD_upper.values(),color = '#D55E00',alpha= 0.1)


ax1.plot(densities,ZL_sep_mean.values(),color = 'y',label = 'ZL-SD')
ax1.plot(densities,ZL_sep_upper.values(),color = 'y',alpha= 0.1)
ax1.plot(densities,ZL_sep_lower.values(),color = 'y',alpha= 0.1)
ax1.fill_between(densities,ZL_sep_lower.values(),ZL_sep_upper.values(),color = 'y',alpha= 0.1)


ax1.set_xlabel('Edges removed/reversed', fontsize = 12)
ax1.grid(True)
ax1.set_ylabel('Average Distance', fontsize = 12)
ax1.legend()
#ax1.set_title('Average Differences of Symmetrized Distance Metrics')


plt.xticks(densities)

plt.yticks(np.arange(0.0,0.8,0.1))

#plt.show()

filename = 'Plots/Differences/'+ 'experiments_several_edge_reversals' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()




