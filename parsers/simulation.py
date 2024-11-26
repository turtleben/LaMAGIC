import argparse
import sys
import os
dir_path = os.getcwd()
sys.path.append(dir_path)

from matplotlib import pyplot as plt
import subprocess
import signal

import random
import numpy as np
import networkx as nx
# import wandb
import torch
from tqdm import tqdm
from threading import Timer

# from transformer_args import get_transformer_args
from parsers.util import *

# sys.path.append(os.path.join(sys.path[0], '../topo_data_util/'))
# from train import main as train_fn
from topo_data_util.topo_analysis.topoGraph import TopoGraph
from parsers.GetReward import calculate_reward
from topo_data_util.topo_utils.plot import plot_hist
from utils.yaml_parser import load_and_apply_yaml_config

simulate_param = {"Duty_Cycle":[0.1, 0.3, 0.5, 0.7, 0.9],
    "Frequency":[1],
    "Rout": [50],
    "Vin": [100],
    "Cout": [10],
    "Ra": [100000],
    "Rb": [1],
    "Rin": [0.1],
    "C": [10],
    "L": [100]
}


def read_LLM_ouput(out_stream):
    vertex_idx = out_stream.find('Vertices')
    edge_idx = out_stream.find('Hyperedges')
    duty_cycle_idx = out_stream.find('The duty cycle is ')
    vertex_string = out_stream[vertex_idx+len('Vertices:'):edge_idx-1]
    # print(vertex_string)
    edge_string = out_stream[edge_idx+len('Hyperedges:'):duty_cycle_idx-1]
    out_strings = out_stream.split()
    if out_strings[-1][-1] == '.':
        duty_cycle = float(out_strings[-1][:len(out_strings[-1])-1])
    else:
        duty_cycle = float(out_strings[-1][:len(out_strings[-1])])
    # print('duty_cycle', duty_cycle)
    # print(edge_string)
    
    start_id = 0
    end_id = 0
    edge_list = []
    for idx, s in enumerate(edge_string):
        if s == '(' or s == '{':
            start_id = idx
        elif s == ')' or s == '}':
            end_id = idx 
            sub_edge_string = edge_string[start_id+1:end_id]
            if sub_edge_string[0] == ' ':
                sub_edge_string = sub_edge_string[1:]
            if sub_edge_string[-1] == ' ':
                sub_edge_string = sub_edge_string[:-1]
            # print('sub_edge_string', sub_edge_string)
            edge = sub_edge_string.split(', ')
            edge_list.append(edge)
    # print(edge_list)
    connect_node_id = 9
    netlist = {}
    for edge in edge_list:
        if 'VIN' in edge:
            parent = 'IN'
        elif 'VOUT' in edge:
            parent = 'OUT'
        elif 'GND' in edge:
            parent = '0'
        else:
            parent = str(connect_node_id)
            connect_node_id += 1
        for node in edge:
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                continue
            if node in netlist.keys():
                netlist[node].append(parent)
            else:
                netlist[node] = [parent]
            
    vertex_list = vertex_string.split(', ')
    # print(netlist)
    # input()
    return netlist, duty_cycle

def read_LLM_output_shrink_canonical(out_stream):
    edge_idx = out_stream.find('Connections')
    duty_cycle_idx = out_stream.find('The duty cycle is ')
    # print(vertex_string)
    edge_string = out_stream[edge_idx+len('Connections:'):duty_cycle_idx-1]
    out_strings = out_stream.split()
    if out_strings[-1][-1] == '.':
        duty_cycle = float(out_strings[-1][:len(out_strings[-1])-1])
    else:
        duty_cycle = float(out_strings[-1][:len(out_strings[-1])])

    edge_list = []
    for idx, s in enumerate(edge_string):
        if s == '(' or s == '{':
            start_id = idx
        elif s == ')' or s == '}':
            end_id = idx 
            sub_edge_string = edge_string[start_id+1:end_id]
            if sub_edge_string[0] == ' ':
                sub_edge_string = sub_edge_string[1:]
            if sub_edge_string[-1] == ' ':
                sub_edge_string = sub_edge_string[:-1]
            # print('sub_edge_string', sub_edge_string)
            edge = sub_edge_string.split(', ')
            edge_list.append(edge)
    connect_node_id = 9
    netlist = {}
    for edge in edge_list:
        if 'VIN' in edge:
            parent = 'IN'
        elif 'VOUT' in edge:
            parent = 'OUT'
        elif 'GND' in edge:
            parent = '0'
        else:
            parent = str(connect_node_id)
            connect_node_id += 1
        for node in edge:
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                continue
            if node in netlist.keys():
                netlist[node].append(parent)
            else:
                netlist[node] = [parent]
    return netlist, duty_cycle

def read_transformer_output_shrink_canonical(out_stream, duty10=False, typeNidx=False):
    # <pad> <duty_0.7> <sep> VIN Sb0 Sb1 Sb2 C0 , VOUT Sb0 Sb1 , GND Sa0 C0 , Sa0 Sb2 , <sep>
    if duty10:
        duty_cycle_map = {'<duty_0.1>': 0.1, '<duty_0.2>': 0.2, '<duty_0.3>': 0.3, '<duty_0.4>': 0.4, '<duty_0.5>': 0.5, '<duty_0.6>': 0.6, '<duty_0.7>': 0.7, '<duty_0.8>': 0.8, '<duty_0.9>': 0.9}
    else:
        duty_cycle_map = {'<duty_0.1>': 0.1, '<duty_0.3>': 0.3, '<duty_0.5>': 0.5, '<duty_0.7>': 0.7, '<duty_0.9>': 0.9}
    output_strings = out_stream.split()
    duty_cycle_token = output_strings[1]
    duty_cycle = duty_cycle_map[duty_cycle_token]
    # print('duty_cycle', duty_cycle)

    out_strings = out_stream.split('<sep>')
    edge_strings = out_strings[1].split(',')
    # print('edge_strings', edge_strings)
    edge_list = []
    if not typeNidx:
        for edge_string in edge_strings:
            edge = edge_string.split()
            edge_list.append(edge)
    else:
        edge = []
        # print('node_list', node_list)
        # print('id_node_map', id_node_map)
        # print('netlist_str', netlist_str)
        for edge_string in edge_strings:
            j = 0
            edge = []
            edge_strs = edge_string.split()
            while j < len(edge_strs):
                node = edge_strs[j]
                if node == "VIN" or node == "VOUT" or node == "GND":
                    edge.append(node)
                else:
                    if j == len(edge_strs) - 1:
                        break
                    node = node + edge_strs[j+1]
                    edge.append(node)
                    j += 1
                j += 1
            edge_list.append(edge)
        # j = 0 
        # netlist_str = out_strings[1].split()
        # print('netlist_str', netlist_str)
        # while j < len(netlist_str):
        #     node = netlist_str[j]
        #     if node == ',':
        #         edge_list.append(edge)
        #         edge = []
        #     elif node == "VIN" or node == "VOUT" or node == "GND":
        #         edge.append(node)
        #     else:
        #         if j == len(netlist_str) - 1:
        #             break
        #         node = node + ' ' + netlist_str[j+1]
        #         edge.append(node)
        #         j += 1
        #     j += 1
        # edge_list.append(edge)
        # print('edge_list', edge_list)

    connect_node_id = 9
    netlist = {}
    for edge in edge_list:
        if 'VIN' in edge:
            parent = 'IN'
        elif 'VOUT' in edge:
            parent = 'OUT'
        elif 'GND' in edge:
            parent = '0'
        else:
            parent = str(connect_node_id)
            connect_node_id += 1
        for node in edge:
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                continue
            if node in netlist.keys():
                netlist[node].append(parent)
            else:
                netlist[node] = [parent]
    return netlist, duty_cycle

def read_transformer_output_mask(vertex_stream, out_stream, duty10=False, pre_eval=False):
    # print(vertex_stream, out_stream)
    # input()
    # <pad> <duty_0.7> <sep> VIN Sb0 Sb1 Sb2 C0 , VOUT Sb0 Sb1 , GND Sa0 C0 , Sa0 Sb2 , <sep>
    if duty10:
        duty_cycle_map = {'<duty_0.1>': 0.1, '<duty_0.2>': 0.2, '<duty_0.3>': 0.3, '<duty_0.4>': 0.4, '<duty_0.5>': 0.5, '<duty_0.6>': 0.6, '<duty_0.7>': 0.7, '<duty_0.8>': 0.8, '<duty_0.9>': 0.9}
    else:
        duty_cycle_map = {'<duty_0.1>': 0.1, '<duty_0.3>': 0.3, '<duty_0.5>': 0.5, '<duty_0.7>': 0.7, '<duty_0.9>': 0.9}
    output_strings = out_stream.split()
    if pre_eval:
        duty_cycle_token = output_strings[0]
        vertex_strings = vertex_stream.split()[:-1]
    else:
        duty_cycle_token = output_strings[1]
        vertex_strings = vertex_stream.split()[:-2]
    duty_cycle = duty_cycle_map[duty_cycle_token]
    # print('duty_cycle', duty_cycle)

    node_idx = 0
    edge_sets = set()
    connect_node_id = 9
    # parent_dict = {'VIN': 'IN', 'VOUT': 'OUT', 'GND': '0'}
    # connection_node_dict = {}
    device_id_dict = {'Sa':0, 'Sb':0, 'L':0, 'C':0}
    for i in range(len(vertex_strings)):
        vertex = vertex_strings[i]
        if vertex == 'VIN' or vertex == 'VOUT' or vertex == 'GND':
            continue
        device_id = device_id_dict[vertex]
        vertex_strings[i] = vertex + str(device_id)
        device_id_dict[vertex] = device_id_dict[vertex] + 1
    # print('vertex_strings', vertex_strings)
    out_strings = out_stream.split('<sep>')
    edge_strings = out_strings[1].split()
    # print(edge_strings)
    idx = 0
    while idx < len(edge_strings):
        # node_name = edge_strings[idx]
        node_name = vertex_strings[node_idx]
        # print(node_name)
        if node_name == 'VIN' or node_name == 'VOUT' or node_name == 'GND':
            for i in range(len(vertex_strings)):
                idx += 1
                continue
        else:
            edge1_list = (node_name,)
            edge2_list = (node_name,)
            for i in range(len(vertex_strings)):
                idx += 1
                if edge_strings[idx] == '<edge_1>' or edge_strings[idx] == '<both_edges>':
                    # print((vertex_strings[i]))
                    edge1_list += (vertex_strings[i],)
                if edge_strings[idx] == '<edge_2>' or edge_strings[idx] == '<both_edges>':
                    edge2_list += (vertex_strings[i],)
            # print('edge1_list', edge1_list)
            # print('edge2_list', edge2_list)
            edge1_list = tuple(sorted(edge1_list))
            edge2_list = tuple(sorted(edge2_list))
            # edge1_list.sort()
            # edge2_list.sort()
            edge_sets.add(edge1_list)
            edge_sets.add(edge2_list)
        node_idx += 1
        idx += 1

    connect_node_id = 9
    netlist = {}
    for edge in edge_sets:
        if 'VIN' in edge:
            parent = 'IN'
        elif 'VOUT' in edge:
            parent = 'OUT'
        elif 'GND' in edge:
            parent = '0'
        else:
            parent = str(connect_node_id)
            connect_node_id += 1
        for node in edge:
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                continue
            if node in netlist.keys():
                netlist[node].append(parent)
            else:
                netlist[node] = [parent]
    # print(netlist)
    # input()
    return netlist, duty_cycle


def read_LLM_output_shrink_canonical_dutycycle(out_stream):
    edge_idx = out_stream.find('Connections')
    duty_cycle_idx = out_stream.find('Duty cycle:')
    # print(vertex_string)
    edge_string = out_stream[edge_idx+len('Connections:'):duty_cycle_idx-1]

    duty_strings = out_stream[duty_cycle_idx:].split()
    
    assert(duty_strings[0] == 'Duty' and duty_strings[1] == 'cycle:')
    duty_cycle_options = [0.1, 0.3, 0.5, 0.7, 0.9]
    duty_strings = duty_strings[2:]
    select_num = 0
    for idx, s in enumerate(duty_strings):
        if s == '<select>':
            duty_cycle = duty_cycle_options[idx]
            select_num += 1
        # elif s != '<unselect>':
    assert(select_num == 1)

    edge_list = []
    for idx, s in enumerate(edge_string):
        if s == '(' or s == '{':
            start_id = idx
        elif s == ')' or s == '}':
            end_id = idx 
            sub_edge_string = edge_string[start_id+1:end_id]
            if sub_edge_string[0] == ' ':
                sub_edge_string = sub_edge_string[1:]
            if sub_edge_string[-1] == ' ':
                sub_edge_string = sub_edge_string[:-1]
            # print('sub_edge_string', sub_edge_string)
            edge = sub_edge_string.split(', ')
            edge_list.append(edge)
    connect_node_id = 9
    netlist = {}
    for edge in edge_list:
        if 'VIN' in edge:
            parent = 'IN'
        elif 'VOUT' in edge:
            parent = 'OUT'
        elif 'GND' in edge:
            parent = '0'
        else:
            parent = str(connect_node_id)
            connect_node_id += 1
        for node in edge:
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                continue
            if node in netlist.keys():
                netlist[node].append(parent)
            else:
                netlist[node] = [parent]
    return netlist, duty_cycle


def read_masked_LLM_output(out_stream, order='duty vertex edge'):
    '''
    Duty cycle: <unselect> <unselect> <select> <unselect> <unselect> <sep> Vertex order: VIN VOUT GND S
    b0 Sb1 C0 L0 L1 <sep> Connections: VIN <no_edge> <no_edge> <no_edge> <no_edge> <edge_1> <no_edge> <no_edge>
    <no_edge> VOUT <no_edge> <no_edge> <no_edge> <no_edge> <no_edge> <no_edge> <no_edge> <edge_1> GND <extra_i
    d_0> Sb0 <no_edge> <no_edge> <no_edge> <no_edge> <edge_1> <edge_1> <edge_2> <no_edge> Sb1 <edge_1> <no_edge
    > <no_edge> <edge_2> <no_edge> <edge_2> <no_edge> <no_edge> C0 <no_edge> <no_edge> <no_edge> <edge_1> <edge
    _1> <no_edge> <no_edge> <edge_2> L0 <no_edge> <no_edge> <edge_1> <edge_2> <no_edge> <no_edge> <no_edge> <no
    _edge> L1 <extra_id_1> <sep> </s></s>
    Duty cycle: <select> <unselect> <unselect> <unselect> <unselect> <sep> Vertex order: VIN VOUT GND Sa0 Sa1 Sb0 L0 L1 <sep> Connections: VIN <no_edge> <no_edge> <no_edge> <edge_1> <no_edge> <no_edge> <edge_1><no_edge> VOUT<no_edge><no_edge><no_edge><no_edge><no_edge><no_edge><no_edge><edge_1> GND<no_edge><no_edge><no_edge><no_edge><no_edge><edge_1><no_edge><no_edge> Sa0<edge_1><no_edge><no_edge><no_edge><edge_2><no_edge><edge_1><edge_2> Sa1<no_edge><no_edge><no_edge><edge_1><no_edge><edge_2><edge_2><edge_1> Sb0<no_edge><no_edge><edge_1><no_edge><edge_2><no_edge><edge_2><no_edge> L0<edge_1><no_edge><no_edge><edge_1><edge_2><edge_2><no_edge><no_edge> L1<no_edge><edge_1><no_edge><edge_2><edge_2><no_edge><no_edge><no_edge><sep>
    '''
    def add_edge(netlist, node, parent):
        if node in netlist.keys():
            if len(netlist[node]) == 2:
                return
            netlist[node].append(parent)
        else:
            netlist[node] = [parent]
    out_strings = out_stream.split('<sep>')
    # print(out_strings)
    if order == 'duty vertex edge':
        duty_strings = out_strings[0].split()
        vertex_strings = out_strings[1].split()
        edge_strings = out_strings[2].split()
    elif order == 'duty edge vertex':
        duty_strings = out_strings[0].split()
        edge_strings = out_strings[1].split()
        vertex_strings = out_strings[2].split()
    elif order == 'vertex edge duty':
        vertex_strings = out_strings[0].split()
        edge_strings = out_strings[1].split()
        duty_strings = out_strings[2].split()
    elif order == 'vertex duty edge':
        vertex_strings = out_strings[0].split()
        duty_strings = out_strings[1].split()
        edge_strings = out_strings[2].split()
    else:
        raise NotImplementedError

    
    assert(duty_strings[0] == 'Duty' and duty_strings[1] == 'cycle:')
    duty_cycle_options = [0.1, 0.3, 0.5, 0.7, 0.9]
    duty_strings = duty_strings[2:]
    select_num = 0
    for idx, s in enumerate(duty_strings):
        if s == '<select>':
            duty_cycle = duty_cycle_options[idx]
            select_num += 1
        # elif s != '<unselect>':
    assert(select_num == 1)
    # print('duty_cycle', duty_cycle)
    
    vertex_strings = vertex_strings[2:]
    node_idx = 0
    edge_sets = set()
    netlist = {}
    connect_node_id = 9
    parent_dict = {'VIN': 'IN', 'VOUT': 'OUT', 'GND': '0'}
    connection_node_dict = {}
    device_id_dict = {'Sa':0, 'Sb':0, 'L':0, 'C':0}
    for i in range(len(vertex_strings)):
        vertex = vertex_strings[i]
        if vertex == 'VIN' or vertex == 'VOUT' or vertex == 'GND':
            continue
        device_id = device_id_dict[vertex]
        vertex_strings[i] = vertex + str(device_id)
        device_id_dict[vertex] = device_id_dict[vertex] + 1
    # print('edge_strings', edge_strings)
    
    idx = 1
    while idx < len(edge_strings):
        # node_name = edge_strings[idx]
        node_name = vertex_strings[node_idx]
        # print(node_name)
        if node_name == 'VIN' or node_name == 'VOUT' or node_name == 'GND':
            for i in range(len(vertex_strings)):
                idx += 1
                continue
        else:
            edge1_list = (node_name,)
            edge2_list = (node_name,)
            for i in range(len(vertex_strings)):
                idx += 1
                if edge_strings[idx] == '<edge_1>' or edge_strings[idx] == '<both_edges>':
                    # print((vertex_strings[i]))
                    edge1_list += (vertex_strings[i],)
                if edge_strings[idx] == '<edge_2>' or edge_strings[idx] == '<both_edges>':
                    edge2_list += (vertex_strings[i],)
            # print(edge1_list)
            edge1_list = tuple(sorted(edge1_list))
            edge2_list = tuple(sorted(edge2_list))
            # edge1_list.sort()
            # edge2_list.sort()
            edge_sets.add(edge1_list)
            edge_sets.add(edge2_list)
        node_idx += 1
        idx += 1

    connect_node_id = 9
    netlist = {}
    for edge in edge_sets:
        if 'VIN' in edge:
            parent = 'IN'
        elif 'VOUT' in edge:
            parent = 'OUT'
        elif 'GND' in edge:
            parent = '0'
        else:
            parent = str(connect_node_id)
            connect_node_id += 1
        for node in edge:
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                continue
            if node in netlist.keys():
                netlist[node].append(parent)
            else:
                netlist[node] = [parent]
    # print(netlist)
    # input()
    return netlist, duty_cycle


    while idx < len(edge_strings):
        node_name = edge_strings[idx]
        # print('node_name', node_name, node_idx)
        if node_name == 'VIN' or node_name == 'VOUT' or node_name == 'GND':
            for i in range(len(vertex_strings)):
                idx += 1
                if edge_strings[idx] == '<edge_1>':
                    node = vertex_strings[i]
                    parent = parent_dict[node_name]
                    # print('1 node', node, 'parent', parent)
                    add_edge(netlist, node, parent)
                    # if node in netlist.keys():
                    #     netlist[node].append(parent)
                    # else:
                    #     netlist[node] = [parent]
        else:
            parent_connection_dict = {}
            for i in range(len(vertex_strings)):
                idx += 1
                if edge_strings[idx] == '<edge_1>' or edge_strings[idx] == '<edge_2>':
                    node = vertex_strings[i]
                    # if i < node_idx and node != 'GND' and node != 'VOUT' and node != 'VIN':
                    #     continue
                    if edge_strings[idx] not in parent_connection_dict.keys():
                        if node == 'VIN' or node == 'VOUT' or node == 'GND':
                            parent = parent_dict[node]
                        else:
                            parent = str(connect_node_id)
                            connect_node_id += 1
                            # print('2 node', node_name, 'parent', parent, i, node_idx)
                            if i >= node_idx:
                                add_edge(netlist, node_name, parent)
                            # add_edge(netlist, node_name, parent)
                        parent_connection_dict[edge_strings[idx]] = parent
                    else:
                        parent = parent_connection_dict[edge_strings[idx]]
                    if not (parent == 'IN' or parent == 'OUT' or parent == '0'):
                        # print('3 node', node, 'parent', parent)
                        if i >= node_idx:
                            add_edge(netlist, node, parent)
                elif edge_strings[idx] == '<both_edges>':
                    node = vertex_strings[i]
                    if node == 'VIN' or node == 'VOUT' or node == 'GND':
                        parent = parent_dict[node]
                    else:
                        parent = str(connect_node_id)
                        connect_node_id += 1
                        if i >= node_idx:
                            add_edge(netlist, node_name, parent)
                    if not (parent == 'IN' or parent == 'OUT' or parent == '0'):
                        if i >= node_idx:
                            add_edge(netlist, node, parent)
                


        node_idx += 1
        idx += 1
    return netlist, duty_cycle

    # for idx in range(2, len(edge_strings)):
def convert_netlist_2_graph(node_tokens, netlist):
    edge_list = []
    node_set = set()
    for node in netlist.keys():
        node_set.add(node)
        connect_nodes = netlist[node]
        for connect_node in connect_nodes:
            edge_list.append((node, connect_node))
            node_set.add(connect_node)
    node_list = list(node_set)
    T = nx.Graph()
    for node in node_list:
        if node in node_tokens:
            if node[0] == 'S':
                T.add_node(node, type=node[:2])
            else:
                T.add_node(node, type=node[:1])
        else:
            T.add_node(node, type='connection')
    T.add_edges_from(edge_list)
    return T
        



def convert_netlist_cki(path, netlist, duty_cycle, L=None):
    file = open(path, 'w')

    # if 'Ra0' in dn:
    #     Ron = pv[dn['Ra0']]
    #     Roff = pv[dn['Rb0']]
    # else:
    Ron = simulate_param["Ra"][0]
    Roff = simulate_param["Rb"][0]

    prefix = [
        ".title buck.cki",
        ".model MOSN NMOS level=8 version=3.3.0",
        ".model MOSP PMOS level=8 version=3.3.0",
        ".model MySwitch SW (Ron=%s.0 Roff=%s.0 vt=%s)" % (Ron, Roff, simulate_param["Vin"][0] / 2),
        ".PARAM vin=%s.0 rin=%s rout=%s.0 cout=%s.0u freq=%s.0M D=%s" % (
            simulate_param["Vin"][0], simulate_param["Rin"][0], simulate_param["Rout"][0], simulate_param["Cout"][0], simulate_param["Frequency"][0], duty_cycle),
        "\n",
        "*input*",
        "Vclock1 gate_a 0 PULSE (0 {vin} 0 1n 1n %su %su)" % (
            1 / simulate_param["Frequency"][0] * duty_cycle, 1 / simulate_param["Frequency"][0]),
        "Vclock2 gate_b 0 PULSE ({vin} 0 0 1n 1n %su %su)" % (
            1 / simulate_param["Frequency"][0] * duty_cycle, 1 / simulate_param["Frequency"][0]),

        "Vin IN_exact 0 dc {vin} ac 1",
        "Rin IN_exact IN {rin}",
        "Rout OUT 0 {rout}",
        "Cout OUT 0 {cout}"
        "\n"]

    sufix = ["\n",
             ".save all",
             # ".save i(vind)",
             ".control",
             # "tran %su 4000u" %(1/simulate_param["Frequency"][0]/10),
             # "tran 1n 2000u",
             "tran 10n 4000u",
             "print V(OUT)",
             "print V(IN_exact,IN)",
             ".endc",
             ".end",
             ]

    file.write("\n".join(prefix) + '\n')
    file.write("*topology*" + '\n')

    line = ''
    for x in netlist:
        if 'S' == x[0]:
            if 'a' == x[1]:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' gate_a gate_b MySwitch'
            elif 'b' == x[1]:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' gate_b gate_a MySwitch'
        elif x[0] == 'C':
            line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' ' + str(simulate_param["C"][0]) + 'u'
        elif x[0] == 'L':
            if L is not None:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' ' + str(L) + 'u'
            else:
                line = x + ' ' + netlist[x][0] + ' ' + netlist[x][1] + ' ' + str(simulate_param["L"][0]) + 'u'
        elif x[0] == 'R':
            if 'a' == x[1]:
                line = x +' '+ netlist[x][0] +' '+ netlist[x][1] +' '+ str(simulate_param["Ra"][0])
            elif 'b' == x[1]:
                line = x +' '+ netlist[x][0] +' '+ netlist[x][1] +' '+ str(simulate_param["Rb"][0])
        else:
            return 0

        line = line + '\n'
        file.write(line)

    file.write("\n".join(sufix) + '\n')
    file.close()
    return

def simulate(path):
    my_timeout = 500
    simu_file = path[:-3] + 'simu'
    try:
        proc = subprocess.Popen('exec ngspice -b ' + path + '>' + simu_file,  shell=True, preexec_fn=os.setsid)
        proc.wait(my_timeout)
    except:
        print("kill\n")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    print('simulation finish')
    return False
    # timer = Timer(my_timeout, proc.kill)
    # try:
    #     timer.start()
    #     stdout, stderr = proc.communicate()
    #     return False
    # finally:
    #     timer.cancel()
    #     print("kill\n")
    #     return True
        
    # p = subprocess.Popen('exec ngspice -b ' + path + '>' + simu_file, stdout=subprocess.PIPE, shell=True)
    # try:
    # p.wait()
    # except subprocess.TimeoutExpired:
    #     print("kill\n")
    #     p.kill()
        

def calculate_efficiency(path, killed):
    simu_file = path[:-3] + 'simu'
    input_voltage = simulate_param["Vin"][0]
    freq = simulate_param["Frequency"][0] * 1000000
    rin = simulate_param["Rin"][0]
    rout = simulate_param["Rout"][0]
    file = open(simu_file, 'r')
    V_in = input_voltage
    V_out = []
    I_in = []
    I_out = []
    time = []

    stable_ratio = 0.01

    cycle = 1 / freq
    # count = 0

    read_V_out, read_I_out, read_I_in = False, False, False
    for line in file:
        # print(line)
        if "Transient solution failed" in line:
            return {'result_valid': False,
                    'efficiency': -1,
                    'Vout': -500,
                    'Iin': -1,
                    'error_msg': 'transient_simulation_failure'}
        if "Index   time            v(out)" in line and not read_V_out:
            read_V_out = True
            read_I_in = False
            continue
        elif "Index   time            v(in_exact,in)" in line and not read_I_in:
            read_V_out = False
            read_I_in = True
            continue

        tokens = line.split()

        # print(tokens)
        if len(tokens) == 3 and tokens[0] != "Index":
            if read_V_out:
                time.append(float(tokens[1]))
                try:
                    V_out.append(float(tokens[2]))
                    I_out.append(float(tokens[2]) / rout)
                except:
                    print('Vout token error')
            elif read_I_in:
                try:
                    I_in.append(float(tokens[2]) / rin)
                except:
                    print('Iin token error')

    print(len(V_out), len(I_in), len(I_out))

    # print(len(V_out),len(I_out),len(I_in),len(time))
    if len(V_out) == len(I_in) == len(I_out) == len(time):
        pass
    else:
        print("don't match")
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'output_is_not_aligned'}

    if not V_out or not I_in or not I_out:
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'missing_output_type'}

    # print(I_out, I_in)
    end = len(V_out) - 1
    start = len(V_out) - 1
    print(cycle, start)
    while start >= 0:
        if time[end] - time[start] >= 50 * cycle:
            break
        start -= 1

    if start == -1:
        print("duration less than one cycle")
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'less_than_one_cycle'}
    mid = int((start + end) / 2)

    # print(start, end,time[end] - time[start])
    P_in = sum([(I_in[x] + I_in[x + 1]) / 2 * (V_in + V_in) / 2 *
                (time[x + 1] - time[x])
                for x in range(start, end)]) / (time[end] - time[start])

    P_out = sum([(I_out[x] + I_out[x + 1]) / 2 * (V_out[x] + V_out[x + 1]) / 2 *
                 (time[x + 1] - time[x])
                 for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave = sum([(V_out[x] + V_out[x + 1]) / 2 * (time[x + 1] - time[x])
                     for x in range(start, end)]) / (time[end] - time[start])

    V_out_ave_1 = np.average(V_out[start:mid])

    V_out_ave_2 = np.average(V_out[mid:end - 1])

    I_in_ave = sum([(I_out[x] + I_out[x + 1]) / 2 * (time[x + 1] - time[x])
                    for x in range(start, end)]) / (time[end] - time[start])

    V_std = np.std(V_out[start:end - 1])

    # print('P_out, P_in', P_out, P_in)
    # if P_in == 0:
    if P_in < 0.001 and P_in > -0.001:
        P_in = 0
        return {'result_valid': False,
                'efficiency': -1,
                'Vout': -500,
                'Iin': -1,
                'error_msg': 'power_in_is_zero'}
    if P_out < 0.001 and P_out > -0.001:
        P_out = 0

    stable_flag = (abs(V_out_ave_1 - V_out_ave_2) <= max(abs(V_out_ave * stable_ratio), V_in / 200))

    # stable_flag = 1;

    eff = P_out / (P_in + 0.01)
    Vout = V_out_ave;
    Iin = I_in_ave;

    result = {'result_valid': (0 <= eff <= 1) and stable_flag,
              'efficiency': eff,
              'Vout': Vout,
              'Iin': Iin,
              'error_msg': 'None'}

    flag_candidate = 0

    if stable_flag == 0:
        result['error_msg'] = 'output_has_not_settled'

    elif eff < 0:
        result['error_msg'] = 'efficiency_is_less_than_zero'

    elif eff > 1:
        result['error_msg'] = 'efficiency_is_greater_than_one'
    elif (V_out_ave < 0.7 * input_voltage or V_out_ave > 1.2 * input_voltage) and eff > 0.7:
        flag_candidate = 1
        print('Promising candidates')

    return result

def sim_generation_output(path, inputs, out_stream_logit, baseline_format='original', llm=None, duty10=False, typeNidx=False):
    """Perform simulation on the out_stream_logit (our circuit description). Path is the path to save simulation results."""
    """Path cannot set to the same when you have multiple experiments running at the same time."""
    if baseline_format == 'original':
        netlist, duty_cycle = read_LLM_ouput(out_stream_logit)
    elif baseline_format == 'shrink_canonical':
        if llm == 'transformer-encoder-decoder':
            netlist, duty_cycle = read_transformer_output_shrink_canonical(out_stream_logit, duty10, typeNidx)
        else:
            netlist, duty_cycle = read_LLM_output_shrink_canonical(out_stream_logit)
    elif baseline_format == 'shrink_canonical_dutycycle':
        netlist, duty_cycle = read_LLM_output_shrink_canonical_dutycycle(out_stream_logit)
    elif baseline_format == 'matrix':
        netlist, duty_cycle = read_transformer_output_mask(inputs, out_stream_logit, duty10)

    else:
        raise NotImplementedError
    # print('read_LLM_ouput')
    # path = "sim.cki"
    convert_netlist_cki(path, netlist, duty_cycle)
    # print('convert_netlist_cki')
    killed = simulate(path)
    # print('simulate')
    result = calculate_efficiency(path, killed)
    return result
    
def sim_masked_generation_output(path, out_stream_logit, order='Duty vertex edge'):
    """Perform simulation on the out_stream_logit (our circuit description). Path is the path to save simulation results."""
    """Path cannot set to the same when you have multiple experiments running at the same time."""
    netlist, duty_cycle = read_masked_LLM_output(out_stream_logit, order=order)
    # print('read_LLM_ouput')
    convert_netlist_cki(path, netlist, duty_cycle)
    # print('convert_netlist_cki')
    killed = simulate(path)
    # print('simulate')
    result = calculate_efficiency(path, killed)
    return result

def sim_netlist_duty_cycle(path, netlist, duty_cycle):
    convert_netlist_cki(path, netlist, duty_cycle)
    # print('convert_netlist_cki')
    killed = simulate(path)
    # print('simulate')
    result = calculate_efficiency(path, killed)
    return result

def sim_netlist_duty_cycle_L(path, netlist, duty_cycle, L):
    convert_netlist_cki(path, netlist, duty_cycle, L)
    # print('convert_netlist_cki')
    killed = simulate(path)
    # print('simulate')
    result = calculate_efficiency(path, killed)
    return result
          
if __name__ == '__main__':
    path = 'example1.cki'
    path = '../try.cki'
    # out_stream = "Here's the circuit representation using a hypergraph:\nVertices:Sa0, Sa1, L0, C0, C1, VIN, VOUT, GND\nHyperedges:{C1, GND}, {C1, VOUT, L0, C0}, {L0, C0, Sa0}, {Sa0, Sa1}, {VIN, Sa1}"
    # out_stream = "Here's the circuit representation using a hypergraph:\nVertices:GND, L1, VOUT, VIN, L0, Sa0, Sb0, Sa1\nHyperedges:(GND, Sb0), (VIN, Sa0, L0), (L1, VOUT), (Sb0, Sa1, L0), (L1, Sa0, Sa1)\nThe duty cycle is set to 0.1."
    
    # The target power conversion ratio is 0.77
    out_stream_logit = "Here's the circuit representation using a hypergraph: Vertices:Sb0, Sa0, Sa1, L0, VIN, Sb1, GND, VOUT Hyperedges:(Sa0, Sb0, Sb1), (Sa1, Sb1, L0), (Sa0, GND), (Sa1, VIN, Sb0), (VOUT, L0) The duty cycle is set to 0.3."
    out_stream_label = "Here's the circuit representation using a hypergraph: Vertices:Sb0, Sa0, Sa1, L0, VIN, Sb1, GND, VOUT Hyperedges:(Sa0, Sb0, Sb1), (Sa1, Sb1, L0), (Sa0, GND), (VOUT, L0, Sa1), (VIN, Sb0) The duty cycle is set to 0.3."
    # The target power conversion ratio is 0.04
    # out_stream_logit = "Here's the circuit representation using a hypergraph: Vertices:C1, VIN, GND, Sa0, C0, VOUT, Sb0, Sb1 Hyperedges:(VOUT, Sb1), (VIN, Sa0, C1), (C1, GND, Sb0), (C0, Sb0), (Sa0, Sb1, C0) The duty cycle is set to 0.9."
    # out_stream_label = "Here's the circuit representation using a hypergraph: Vertices:C1, VIN, GND, Sa0, C0, VOUT, Sb0, Sb1 Hyperedges:(VOUT, Sb1), (VIN, Sa0, C0, C1), (Sb0, C1), (Sa0, Sb1, Sb0), (C0, GND) The duty cycle is set to 0.9."
    
    # out_stream_logit = "Here's the circuit representation using a hypergraph: Vertices:GND, VOUT, VIN, L0, C0, Sa0, Sb0, Sa1 Hyperedges:(Sb0, VOUT, C0, Sa1, L0), (Sa0, GND, Sb1), (Sb1, VIN, Sb0), (C0, Sa0, L0, Sa1) The duty cycle is set to 0.7."
    
    out_stream_logit = "Here's the circuit representation using a hypergraph: Vertices:GND, Sa0, Sb0, Sb1, L0, C0, VIN, VOUT Hyperedges:(VIN, L0), (L0, Sa0), (Sa0, Sb0, Sb1, C0), (C0, VOUT), (C0, GND), (Sb0, Sb1, GND) The duty cycle is set to 0.3."
    
    result = sim_generation_output(path, None,  out_stream_logit)
    print('result', result)
    
    # args = get_transformer_args()
    # config_path = 'parser/config/parser.yaml'
    # config = load_and_apply_yaml_config(config_path)
    # os.makedirs(config.text_data_dir, exist_ok=True))