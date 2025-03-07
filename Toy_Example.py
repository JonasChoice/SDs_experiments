import numpy as np

import sys



from codebase import metrics
from codebase import mixed_graph as mixed

SHD_dict = {}

parent_AID_dict = {}
ancestor_AID_dict = {}

parent_SD_dict = {}
ancestor_SD_dict = {}
ZL_SD_dict = {}

SHD_dict_inverse = {}

parent_AID_dict_inverse = {}
ancestor_AID_dict_inverse = {}

parent_SD_dict_inverse = {}
ancestor_SD_dict_inverse = {}
ZL_SD_dict_inverse = {}

parent_AID_dict_CPDAG = {}
ancestor_AID_dict_CPDAG = {}

parent_AID_dict_CPDAG_inverse = {}
ancestor_AID_dict_CPDAG_inverse = {}

pparent_SD_dict_CPDAG = {}
pparent_SD_dict_CPDAG_inverse = {}

SHD_CPDAG_dict = {}

M_max = 21

for M in range(3,M_max):

    '''Generate the graphs G and H'''
    G = mixed.LabelledMixedGraph()
    H = mixed.LabelledMixedGraph()
    nodes = []
    for i in range(1,2*M+2):
        nodes.append('X'+str(i))

    for node in nodes:
        G.add_node(node)
        H.add_node(node)

    for i in range(1, 2 * M+1):
        G.add_directed('X'+str(i), 'X'+str(i+1))

    for i in range(1, M+1):
        H.add_directed('X' + str(2*i-1), 'X' + str(2*i + 1))

    H.add_directed('X' + str(2 * M+1), 'X' + str(2))

    for i in range(1, M):
        H.add_directed('X' + str(2 * i), 'X' + str(2 * i + 2))


    #print(G)
    #print(H)
    SHD_dict[M] = metrics.SHD_DAGs(G,H)
    parent_AID_dict[M] = metrics.parent_AID_DAGs(G,H)
    #ancestor_AID_dict[M] = metrics.ancestor_AID_DAGs(G,H)

    parent_SD_dict[M] = metrics.SD_DAGs(G,H,type='parent')
    #ancestor_SD_dict[M] = metrics.SD(G,H,type='ancestor')
    #ZL_SD_dict[M] = metrics.SD(G,H,type='ZL')

    SHD_dict_inverse[M] = metrics.SHD_DAGs(H, G)
    parent_AID_dict_inverse[M] = metrics.parent_AID_DAGs(H, G)
    #ancestor_AID_dict_inverse[M] = metrics.ancestor_AID_DAGs(H, G)

    parent_SD_dict_inverse[M] = metrics.SD_DAGs(H, G, type='parent')
    #ancestor_SD_dict_inverse[M] = metrics.SD(H, G, type='ancestor')
    #ZL_SD_dict_inverse[M] = metrics.SD(H, G, type='ZL')


#print(SHD_dict,parent_AID_dict,ancestor_AID_dict,parent_SD_dict,ancestor_SD_dict,ZL_SD_dict)
#print(parent_AID_dict)

    '''Comparison on CPDAGs'''
    #change this!
    #adj_CPDAG_G = np.zeros(shape=(2*M+1,2*M+1),dtype=np.int8)
    #adj_CPDAG_H = np.zeros(shape=(2 * M + 1, 2 * M + 1), dtype=np.int8)

    #for i in range(2 * M):
    #    adj_CPDAG_G[i,i+1] = 2

    #for i in range(M):
    #    adj_CPDAG_H[2*i, 2*(i + 1)] = 2

    #adj_CPDAG_H[1, 2 * M] = 2

    #for i in range(M - 1):
    #    adj_CPDAG_H[2 * i +1 , 2 * i + 3] = 2

    CPDAG_G = G.get_CPDAG()
    CPDAG_H = H.get_CPDAG()
    #if M == 3:
    #    print(CPDAG_G,CPDAG_H)
    #print(CPDAG_G,CPDAG_H)
    #print(adj_CPDAG_G)
    #adj_CPDAG_G[np.where(adj_CPDAG_G == 1)] = 2

    #print(adj_CPDAG_G)
    #print(adj_CPDAG_H)

    SHD_CPDAG_dict[M] = metrics.SHD_CPDAGs(CPDAG_G,CPDAG_H,normalized = True)
    pparent_SD_dict_CPDAG[M] = metrics.SD_CPDAGs(CPDAG_G,CPDAG_H,type = 'pparent', normalized = True)
    print(pparent_SD_dict_CPDAG[M])
    parent_AID_dict_CPDAG[M] = metrics.parent_AID_CPDAGs(CPDAG_G,CPDAG_H,normalized = True)
    #print(parent_AID_dict_CPDAG[M])

    #ancestor_AID_dict_CPDAG[M] = gadjid.ancestor_aid(adj_CPDAG_G, adj_CPDAG_H)[0]
    parent_AID_dict_CPDAG_inverse[M] = metrics.parent_AID_CPDAGs(CPDAG_H,CPDAG_G,normalized = True)
    #ancestor_AID_dict_CPDAG_inverse[M] = metrics.parent_AID_CPDAGs(CPDAG_H,CPDAG_G,normalized = True)
    pparent_SD_dict_CPDAG_inverse[M] = metrics.SD_CPDAGs(CPDAG_H, CPDAG_G, type='pparent', normalized=True)

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('seaborn-v0_8-colorblind')

fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2,2,sharex=True, sharey=True)

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)

#ax1.plot(SHD_dict.keys(),SHD_dict.values(),label = 'SHD')
ax1.plot(parent_AID_dict.keys(),parent_AID_dict.values(),label = 'parentAID')
#ax1.plot(ancestor_AID_dict.keys(),ancestor_AID_dict.values(),label = 'ancestorAID')
ax1.plot(parent_SD_dict.keys(),parent_SD_dict.values(),label = 'parentSD')
#ax1.plot(ancestor_SD_dict.keys(),ancestor_SD_dict.values(),label = 'ancestorSD')
#ax1.plot(ZL_SD_dict.keys(),ZL_SD_dict.values(),label = 'ZL-SD')

#ax2.plot(SHD_dict_inverse.keys(),SHD_dict_inverse.values(),label = 'SHD')
ax2.plot(parent_AID_dict_inverse.keys(),parent_AID_dict_inverse.values(),label = 'parentAID')
#ax2.plot(ancestor_AID_dict_inverse.keys(),ancestor_AID_dict_inverse.values(),label = 'ancestorAID')
ax2.plot(parent_SD_dict_inverse.keys(),parent_SD_dict_inverse.values(),label = 'parentSD')
#ax2.plot(ancestor_SD_dict_inverse.keys(),ancestor_SD_dict_inverse.values(),label = 'ancestorSD')
#ax2.plot(ZL_SD_dict_inverse.keys(),ZL_SD_dict_inverse.values(),label = 'ZL-SD')


#ax1.set_xlabel('M', fontsize = 12)
#ax2.set_xlabel('M', fontsize = 12)
ax1.set_ylabel('DAGs', fontsize = 12)
ax1.legend(loc='upper left', fontsize = 8)
ax1.set_title('d(G,H)')
ax2.set_title('d(H,G)')

plt.xticks(range(2,M_max,2))

#plt.show()

#filename = 'Plots/Toy_example_2/'+ 'distances_comparison_all' + '.png'
#plt.savefig(filename, bbox_inches="tight")
#plt.close()

#fig2, (ax3,ax4) = plt.subplots(1,2,sharex=True, sharey=True)

#ax3.plot(SHD_CPDAG_dict.keys(),SHD_CPDAG_dict.values(),label = 'SHD')
ax3.plot(parent_AID_dict_CPDAG.keys(),parent_AID_dict_CPDAG.values(),label = 'parentAID')
#ax3.plot(ancestor_AID_dict_CPDAG.keys(),ancestor_AID_dict_CPDAG.values(),label = 'ancestorAID')
ax3.plot(pparent_SD_dict_CPDAG.keys(),pparent_SD_dict_CPDAG.values(),label = 'pparentSD')
#ax3.plot(ancestor_SD_dict.keys(),ancestor_SD_dict.values(),label = 'ancestorSD')
#ax3.plot(ZL_SD_dict.keys(),ZL_SD_dict.values(),label = 'ZL-SD')

#ax4.plot(SHD_CPDAG_dict.keys(),SHD_CPDAG_dict.values(),label = 'SHD')
ax4.plot(parent_AID_dict_CPDAG_inverse.keys(),parent_AID_dict_CPDAG_inverse.values(),label = 'parentAID')
#ax4.plot(ancestor_AID_dict_CPDAG_inverse.keys(),ancestor_AID_dict_CPDAG_inverse.values(),label = 'ancestorAID')
ax4.plot(pparent_SD_dict_CPDAG_inverse.keys(),pparent_SD_dict_CPDAG_inverse.values(),label = 'pparentSD')
#ax4.plot(ancestor_SD_dict_inverse.keys(),ancestor_SD_dict_inverse.values(),label = 'ancestorSD')
#ax4.plot(ZL_SD_dict_inverse.keys(),ZL_SD_dict_inverse.values(),label = 'ZL-SD')

ax3.set_xlabel('M', fontsize = 12)
ax4.set_xlabel('M', fontsize = 12)
ax3.set_ylabel('CPDAGs', fontsize = 12)
ax3.legend(loc='upper left',fontsize = 8)
#ax3.set_title('d(CPDAG(G),CPDAG(H))')
#ax4.set_title('d(CPDAG(H),CPDAG(G))')


plt.xticks(range(2,M_max,2))
plt.yticks(np.arange(0.0,1.1,0.2))

#plt.show()

filename = 'Plots/Toy_example/'+ 'distances_comparison_all_in_one_plot' + '.png'
plt.savefig(filename, bbox_inches="tight")
plt.close()