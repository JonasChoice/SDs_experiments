import copy

import numpy as np

import sys


from codebase import mixed_graph as mixed
import itertools


def generate_causal_order(N,random_state=None):
    ## generates a random causal order for data generation
    if random_state is None:
        random_state = np.random
    causal_order = list(random_state.permutation(N))
    return causal_order


def generate_ER_random_DAG(N,p,random_state = None):

    if random_state is None:
        random_state = np.random

    help_list = list(random_state.permutation(N))
    causal_order = ['X'+str(i) for i in help_list]

    G = mixed.LabelledMixedGraph(nodes = causal_order)

    for node in causal_order:
        for node2 in causal_order[causal_order.index(node)+1:]:
            if random_state.binomial(1,p) == 1:
                G.add_directed(node, node2)

    return G


def generate_empty_graph(N):

    causal_order = ['X'+str(i) for i in range(N)]

    G = mixed.LabelledMixedGraph(nodes = causal_order)

    return G

def generate_ER_random_MAG(N,p,number_of_latents, random_state = None):

    if random_state is None:
        random_state = np.random


    def powerset(s, r_min=0, r_max=None):
        if r_max is None: r_max = len(s)
        return map(set, itertools.chain(*(itertools.combinations(s, r) for r in range(r_min, r_max + 1))))

    DAG = generate_ER_random_DAG(N,p,random_state)

    observed = copy.deepcopy(DAG._nodes)
    latent_nodes = []

    for i in range(number_of_latents):
        latent_nodes.append('L'+str(i))

    #extended_graph = copy.deepcopy(DAG)

    for i in latent_nodes:
        DAG.add_node(i)

        targets = random_state.choice(list(observed),2,replace = False)
        for j in targets:
            DAG.add_directed(i,j)

    directed = set()
    bidirected = set()

    for i, j in itertools.combinations(observed, r=2):
        adjacent = all(not DAG.is_d_separated(i, j, S) for S in powerset(observed - {i, j}))
        if adjacent:
            if DAG.is_ancestor_of(i, j):
                directed.add((i, j))
            elif DAG.is_ancestor_of(j, i):
                directed.add((j, i))
            else:
                bidirected.add((i, j))

    directed_with_label = {(i,j): None for (i,j) in directed}
    bidirected_with_label = {frozenset({i,j}): None for (i,j) in bidirected}

    return mixed.LabelledMixedGraph(nodes=observed, directed=directed_with_label, bidirected=bidirected_with_label)


def generate_ER_random_mixed(N,p,prob_bidirected,random_state = None):

    if random_state is None:
        random_state = np.random

    help_list = list(random_state.permutation(N))
    causal_order = ['X'+str(i) for i in help_list]

    G = mixed.LabelledMixedGraph(nodes = causal_order)

    for node in causal_order:
        for node2 in causal_order[causal_order.index(node)+1:]:
            if random_state.binomial(1,p) == 1:
                if random_state.binomial(1, prob_bidirected) == 1:
                    G.add_bidirected(node, node2)
                else:
                    G.add_directed(node, node2)

    return G

# def ZL_SD_for_MAG_experiments(graph1, graph2, normalized = True):
#     if isinstance(graph1, mixed.LabelledMixedGraph):
#         G_1 = graph1
#     else:
#         G_1 = metrics.string_graph_to_mixed_graph(graph1)
#     if isinstance(graph2, mixed.LabelledMixedGraph):
#         G_2 = graph2
#     else:
#         G_2 = metrics.string_graph_to_mixed_graph(graph2)
#
#     if G_1.nodes != G_2.nodes:
#         raise ValueError('graphs have different nodes!')
#
#     variables = G_1.nodes
#     N = len(variables)
#
#     if type == 'ZL':
#         '''computation of the van der Zander-Liskiewicz separation distance for mixed graphs'''
#         canonical_DAG_1 = G_1.get_canonical_directed_graph()
#
#         canonical_DAG_2 = G_2.get_canonical_directed_graph()
#
#
#         pairs = list(itertools.product(G_2.nodes, G_2.nodes))
#         for node in G_2.nodes:
#             pairs.remove((node, node))
#
#         for (node1, node2) in pairs:
#             if not ((node1, node2) in G_2.directed_keys or (node2, node1) in G_2.directed_keys or frozenset(
#                     {node2, node1}) in G_2.bidirected_keys):
#
#                 sep = canonical_DAG_2.find_minimal_d_separator({node1}, {node2},restricted=G_2.nodes,DAG_check=False)
#                 if not sep is None:
#                     separable_node_pairs[(node1, node2)] = sep
#
#         error_count = 0
#
#         for (X, Y) in separable_node_pairs.keys():
#
#             # print({X},{Y},separable_node_pairs[(X,Y)])
#             # print(G_2)
#             if not canonical_DAG_1.is_d_separated({X}, {Y}, set(separable_node_pairs[(X, Y)]), DAG_check=False):
#                 error_count += 1
#
#         if normalized == True:
#             return error_count / (N * (N - 1))
#         else:
#             return error_count

def adjacency_to_graph(nodes,matrix):
    G = mixed.LabelledMixedGraph(nodes = nodes)
    for i in nodes:
        for j in nodes[nodes.index(i)+1:]:
            if matrix[nodes.index(i),nodes.index(j)] == 1 and matrix[nodes.index(j),nodes.index(i)] == 1:
                G.add_undirected(i,j)
            if matrix[nodes.index(i),nodes.index(j)] == 1 and matrix[nodes.index(j),nodes.index(i)] == 0:
                G.add_directed(i,j)
            if matrix[nodes.index(i),nodes.index(j)] == 0 and matrix[nodes.index(j),nodes.index(i)] == 1:
                G.add_directed(j,i)

    return G

def string_graph_to_mixed_graph(graph):
    G = mixed.LabelledMixedGraph()
    for (i, j,t) in zip(*np.where(graph != '')):
        if graph[i, j,t] == '-->':
            G.add_directed('X'+str(i)+'_'+str(t),'X'+str(j)+'_'+str(t))
        if graph[i, j,t] == '<->':
            G.add_bidirected('X'+str(i)+'_'+str(t),'X'+str(j)+'_'+str(t))
        if graph[i, j,t] == '---':
            G.add_undirected('X'+str(i)+'_'+str(t),'X'+str(j)+'_'+str(t))
    return G
