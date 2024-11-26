import collections
from shutil import copy
import os


def graph_to_adjacency_matrix(graph, node_list):
    """
    :param graph: {node: {neighbors}}
    :return: [i][j] == 1 if i-th and j-th nodes are connected in graph
    """
    indexed_graph = {node_list.index(node): [node_list.index(neighbor) for neighbor in neighbors]
                     for node, neighbors in graph.items()}

    return indexed_graph_to_adjacency_matrix(indexed_graph)

def indexed_graph_to_adjacency_matrix(graph):
    node_num = len(graph)

    adjacent_matrix = [[0 for i in range(node_num)] for j in range(node_num)]

    for i in graph.keys():
        for j in graph[i]:
            adjacent_matrix[i][j] = 1
            adjacent_matrix[j][i] = 1

    return adjacent_matrix

def adj_matrix_to_graph(node_list, adj_matrix):
    node_num = len(node_list)

    graph = collections.defaultdict(list)

    for i in range(node_num):
        for j in range(node_num):
            if adj_matrix[i][j] == 1:
                graph[node_list[i]].append(node_list[j])

    return dict(graph)

def nodes_and_edges_to_adjacency_matrix(node_list, edge_list):
    node_num = len(node_list)

    adjacent_matrix = [[0 for i in range(node_num)] for j in range(node_num)]

    for edge in edge_list:
        # for node in edge:
        i = node_list.index(edge[0])
        j = node_list.index(edge[1])
        adjacent_matrix[i][j] = 1
        adjacent_matrix[j][i] = 1

    return adjacent_matrix

def adj_matrix_to_edges(node_list, adj_matrix):
    edges = []
    for node_i in node_list:
        for node_j in node_list:
            if adj_matrix[node_list.index(node_i)][node_list.index(node_j)]:
                edges.append((node_i, node_j))

    return edges
