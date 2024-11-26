import argparse
import sys
import os
import copy
dir_path = os.getcwd()
sys.path.append(dir_path)

import itertools
import json
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
import networkx as nx
import random
import numpy as np
from tqdm import tqdm
from transformer_args import get_transformer_args
from parsers.simulation import sim_generation_output, read_masked_LLM_output, convert_netlist_2_graph, sim_netlist_duty_cycle, read_LLM_ouput, read_LLM_output_shrink_canonical, read_LLM_output_shrink_canonical_dutycycle, read_transformer_output_shrink_canonical, read_transformer_output_mask
from topo_data_util.topo_analysis.topoGraph import TopoGraph
from analog_LLM.utils.utils import combine_masked_input_output

def sim_cir(datum):
    path = 'sim_check.cki'
    instruction, input_ids, output = gen_textdata_from_raw(datum)
    return sim_generation_output(path, output), output


def gen_textdata_from_raw(datum):
    
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    phase_one_switches = []
    phase_two_switches = []
    capacitances = []
    inductors = []
    ports = []
    input_ids = ""
    output = "Here's the circuit representation using a hypergraph:\nVertices:"
    
    for i, node in enumerate(graph.node_list):
        if type(node) == int:
            continue
        output += "{}".format(node)
        if i != len(graph.node_list) - 1:
            output += ', '
        
        if node.startswith('Sa'):
            phase_one_switches.append(node)
        elif node.startswith('Sb'):
            phase_two_switches.append(node)
        elif node.startswith('L'):
            inductors.append(node)
        elif node.startswith('C'):
            capacitances.append(node)
        elif node.startswith('V') or node.startswith('G'):
            ports.append(node)
        else:
            raise NotImplementedError
    output +=  "\nHyperedges:"
    for i, edge in enumerate(graph.hyper_edge_list):
        output += str(edge)
        if i != len(graph.hyper_edge_list) - 1:
            output += ', '
    output += "\nThe duty cycle is set to {}.".format(datum["duty_cycle"])
        
    # instruction = "Given "
    # input = 'This circuit has '
    
    def gen_string(devices, name):
        instruction = ""
        if len(devices) > 0:
            instruction += "{} {}".format(len(devices), name)
            if len(devices) > 1:
                if name.endswith('switch'):
                    instruction += 'es'
                else:
                    instruction += 's'
            instruction += ' '
            for i, s in enumerate(devices):
                if i > 0:
                    instruction += " and "
                instruction += "{}".format(s)
            instruction += ", "
        return instruction
            
    # instruction += gen_string(phase_one_switches, "phase-one switch")
    # instruction += gen_string(phase_two_switches, "phase-two switch")
    # instruction += gen_string(inductors, "inductor")
    # instruction += gen_string(capacitances, "capacitance")
    # instruction += "a circuit input VIN, a circuit output VOUT, a ground GND, "
    
    input_ids += gen_string(phase_one_switches, "phase-one switch")
    input_ids += gen_string(phase_two_switches, "phase-two switch")
    input_ids += gen_string(inductors, "inductor")
    input_ids += gen_string(capacitances, "capacitance")
    input_ids += "a circuit input VIN, a circuit output VOUT, a ground GND. "
    input_ids += "The duty cycle has five options (0.1, 0.3, 0.5, 0.7, 0.9). "
    # for i, port in enumerate(ports):
    #     if i == len(port) - 1:
    #         input_ids += " and"
    #     input_ids += "{}, ".format(port)
    # instruction += "by setting duty cycle to {}, ".format(datum["duty_cycle"])
    vout = datum["vout"] / 100
    input_ids += "The target voltage conversion ratio is {:.2f}.".format(vout)
    # torch.clamp(vout / 100., 0., 1.)
    
    # instruction += 'generate a circuit topology and select the duty cycle with options {{0.1, 0.3, 0.5, 0.7, 0.9}} to achieve a target power conversion ratio of {:.2f}'.format(vout)
    # instruction = 'Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve a target power conversion ratio of {:.2f}.'.format(vout)
    instruction = 'Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve the following target voltage conversion ratio.'
    
    return instruction, input_ids, output

  
def gen_textdata_topo2power_from_raw(datum):
    
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    phase_one_switches = []
    phase_two_switches = []
    capacitances = []
    inductors = []
    ports = []
    hgraph = "the circuit representation using a hypergraph is:\nVertices:"
    input_ids = ""
    
    for i, node in enumerate(graph.node_list):
        if type(node) == int:
            continue
        hgraph += "{}".format(node)
        if i != len(graph.node_list) - 1:
            hgraph += ', '
        if node.startswith('Sa'):
            phase_one_switches.append(node)
        elif node.startswith('Sb'):
            phase_two_switches.append(node)
        elif node.startswith('L'):
            inductors.append(node)
        elif node.startswith('C'):
            capacitances.append(node)
        elif node.startswith('V') or node.startswith('G'):
            ports.append(node)
        else:
            raise NotImplementedError
    hgraph +=  "\nHyperedges:"
    for i, edge in enumerate(graph.hyper_edge_list):
        hgraph += str(edge)
        if i != len(graph.hyper_edge_list) - 1:
            hgraph += ', '
    # hgraph += "\nThe duty cycle is set to {}.".format(datum["duty_cycle"])
        
    # instruction = "Given a circuit with "
    input_ids = "This circuit has "
    
    def gen_string(devices, name):
        instruction = ""
        if len(devices) > 0:
            instruction += "{} {}".format(len(devices), name)
            if len(devices) > 1:
                if name.endswith('switch'):
                    instruction += 'es'
                else:
                    instruction += 's'
            instruction += ' '
            for i, s in enumerate(devices):
                if i > 0:
                    instruction += " and "
                instruction += "{}".format(s)
            instruction += ", "
        return instruction
            
    # instruction += gen_string(phase_one_switches, "phase-one switch")
    # instruction += gen_string(phase_two_switches, "phase-two switch")
    # instruction += gen_string(inductors, "inductor")
    # instruction += gen_string(capacitances, "capacitance")
    # instruction += "a circuit input VIN, a circuit output VOUT, a ground GND, "
    # instruction += "and " + hgraph + '\n'
    
    input_ids += gen_string(phase_one_switches, "phase-one switch")
    input_ids += gen_string(phase_two_switches, "phase-two switch")
    input_ids += gen_string(inductors, "inductor")
    input_ids += gen_string(capacitances, "capacitance")
    input_ids += "a circuit input VIN, a circuit output VOUT, a ground GND, "
    input_ids += "and " + hgraph + '\n'
    input_ids += 'The duty cycle is {:.2f}.'.format(datum["duty_cycle"])
    
    vout = datum["vout"] / 100
    # torch.clamp(vout / 100., 0., 1.)
    # instruction += 'By setting duty cycle to {:.2f}, '.format(datum["duty_cycle"])
    # instruction += 'evaluate its power conversion ratio.'
    instruction = 'Evaluate the voltage conversion ratio of the following cirucit and duty cycle.'
    # output = "The power conversion ratio is {:.2f}.".format(vout)
    output = "{:.2f}.".format(vout)
    
    return instruction, input_ids, output

def sim_single_data(datum, prefix=None):
    if prefix is None:
        prefix = path = 'sim_check.cki'
    else:
        path = os.path.join(prefix, 'sim_check.cki')
    instruction, input_ids, output = gen_textdata_from_raw(datum)
    print(output)
    result = sim_generation_output(path, output)
    return result['Vout'], result['efficiency']

def gen_adjmatrix_textdata_from_raw(datum):
    # print(datum)
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.hyper_edges2adj_matrix()
    # print(graph.adj_matrix)
    d_dict = {}
    d_dict['vout'] = datum["vout"] / 100
    d_dict['eff'] = datum["eff"]
    node_order_str = "Vertex order:"
    for i, node in enumerate(graph.node_list):
        node_order_str += node
        if i != len(graph.node_list) - 1:
            node_order_str += ' '
    node_order_str += '<sep>'
    # print('node_order_str', node_order_str)
    edge_matrix_str = "Connections:"
    for i, node in enumerate(graph.node_list):
        edge_matrix_str += (node + '')
        for e in graph.adj_matrix[i]:
            if int(e) == 0:
                edge_matrix_str += '<no_edge>'
            elif int(e) == 1 or int(e) == 2:
                edge_matrix_str += '<edge_{}>'.format(int(e))
            else:
                raise NotImplementedError
        # edge_matrix_str += ' '.join(str(int(e)) for e in graph.adj_matrix[i])
        # edge_matrix_str += ' '
        if i != len(graph.node_list) - 1:
            edge_matrix_str += ' '
        else:
            edge_matrix_str += '<sep>'
    duty_cycle_order = {0.1:0, 0.3:1, 0.5:2, 0.7:3, 0.9:4}
    duty_one_hot = np.zeros((5))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_one_str = "Duty cycle:"
    for e in duty_one_hot:
        if int(e) == 0:
            duty_one_str += '<unselect>'
        elif int(e) == 1:
            duty_one_str += '<select>'
        else:
            raise NotImplementedError
    duty_one_str += '<sep>'
    # duty_one_str += ' '.join(str(int(e)) for e in duty_one_hot)
    # print('duty_one_hot', duty_one_hot)
    # print('edge_matrix_str', edge_matrix_str)
    # print('duty_one_str', duty_one_str)

    circuit_str = duty_one_str + node_order_str + edge_matrix_str
    # print('circuit_str', circuit_str)
    # input()
    d_dict['circuit_str'] = circuit_str
    d_dict['list_of_node'] = datum['list_of_node']
    d_dict['list_of_edge'] = datum['list_of_edge']
    # if len(graph.node_list) == 6:
    print(circuit_str)
    input()
    return d_dict

def gen_adjmatrix_textdata_from_raw_for_test(datum):
    # print(datum)
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.hyper_edges2adj_matrix()
    # print(graph.adj_matrix)
    d_dict = {}
    d_dict['vout'] = datum["vout"] / 100
    d_dict['eff'] = datum["eff"]
    node_order_str = "Vertex order: "
    for i, node in enumerate(graph.node_list):
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            node_order_str += node
        else:
            node_order_str += node[:-1]
        
        if i != len(graph.node_list) - 1:
            node_order_str += ' '
    node_order_str += ' <sep> '
    # print('node_order_str', node_order_str)
    edge_matrix_str = "Connections: "
    for i, node in enumerate(graph.node_list):
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            edge_matrix_str += (node + ' ')
        else:
            edge_matrix_str += (node[:-1] + ' ')
        for e in graph.adj_matrix[i]:
            if int(e) == 0:
                edge_matrix_str += '<no_edge> '
            elif int(e) == 1 or int(e) == 2:
                edge_matrix_str += '<edge_{}> '.format(int(e))
            elif int(e) == 3:
                edge_matrix_str += '<both_edges> '
            else:
                raise NotImplementedError
        # edge_matrix_str += ' '.join(str(int(e)) for e in graph.adj_matrix[i])
        # edge_matrix_str += ' '
        if i != len(graph.node_list) - 1:
            edge_matrix_str += ''
        else:
            edge_matrix_str += '<sep> '
    duty_cycle_order = {0.1:0, 0.3:1, 0.5:2, 0.7:3, 0.9:4}
    duty_one_hot = np.zeros((5))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_one_str = "Duty cycle: "
    for e in duty_one_hot:
        if int(e) == 0:
            duty_one_str += '<unselect> '
        elif int(e) == 1:
            duty_one_str += '<select> '
        else:
            raise NotImplementedError
    duty_one_str += '<sep> '
    # duty_one_str += ' '.join(str(int(e)) for e in duty_one_hot)
    # print('duty_one_hot', duty_one_hot)
    # print('edge_matrix_str', edge_matrix_str)
    # print('duty_one_str', duty_one_str)

    '''Version 1 duty cycle then graph'''
    circuit_str = duty_one_str + node_order_str + edge_matrix_str
    '''Version 2 graph then duty cycle'''
    # circuit_str = node_order_str + edge_matrix_str + duty_one_str
    # print('circuit_str', circuit_str)
    # input()
    d_dict['circuit_str'] = circuit_str
    d_dict['list_of_node'] = datum['list_of_node']
    d_dict['list_of_edge'] = datum['list_of_edge']
    # if len(graph.node_list) == 6:
    # print(circuit_str)
    return d_dict

def convert_raw_2_matrix(raw_data):
    vouts = []
    effs = []
    for datum in tqdm(raw_data):
        # print(datum)
        vouts.append(datum['vout'])
        effs.append(datum['eff'])
    print(np.max(effs), np.min(effs))
    vouts = np.array(vouts)
    effs = np.array(effs)
    upper_threshold_power = np.percentile(vouts, 99.5)
    lower_threshold_power = np.percentile(vouts, 0.5)
    # upper_threshold_eff = np.percentile(effs, 99.5)
    upper_threshold_eff = 1.0
    lower_threshold_eff = 0.0
    # lower_threshold_eff = np.percentile(effs, 0.5)
    print('lower_threshold_eff', lower_threshold_eff)
    print('upper_threshold_eff', upper_threshold_eff)

    n_invalid = 0
    matrix_data = []
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            if datum["vout"] != -500:
                continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            if datum["eff"] != -1:
                # vout, eff = sim_single_data(datum)
                # print('original vout and eff: ', datum["vout"], ', ', datum["eff"])
                # print('simulate vout and eff: ', vout, ', ', eff)
                # print(datum)
                # input()
                continue
        if datum["vout"] == -500 and datum["eff"] == -1:
            n_invalid += 1
        # d_dict = gen_adjmatrix_textdata_from_raw(datum)
        d_dict = gen_adjmatrix_textdata_from_raw_for_test(datum)
        matrix_data.append(d_dict)
    print('n_invalid', n_invalid)
    print('len(matrix_data)', len(matrix_data))
    return matrix_data

def gen_adjmatrix_textdata_from_raw_input_labels(datum):
    # print(datum)
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.hyper_edges2adj_matrix()
    # print(graph.adj_matrix)
    d_dict = {}
    d_dict['vout'] = datum["vout"] / 100
    d_dict['eff'] = datum["eff"]
    node_order_str = "Vertex order: "
    for i, node in enumerate(graph.node_list):
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            node_order_str += node
        else:
            node_order_str += node[:-1]
        
        if i != len(graph.node_list) - 1:
            node_order_str += ' '
    node_order_str += ' <sep> '
    # print('node_order_str', node_order_str)
    edge_matrix_str = "Connections: "
    for i, node in enumerate(graph.node_list):
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            edge_matrix_str += (node + ' ')
        else:
            edge_matrix_str += (node[:-1] + ' ')
        for e in graph.adj_matrix[i]:
            if int(e) == 0:
                edge_matrix_str += '<no_edge> '
            elif int(e) == 1 or int(e) == 2:
                edge_matrix_str += '<edge_{}> '.format(int(e))
            elif int(e) == 3:
                edge_matrix_str += '<both_edges> '
            else:
                raise NotImplementedError
        # edge_matrix_str += ' '.join(str(int(e)) for e in graph.adj_matrix[i])
        # edge_matrix_str += ' '
        if i != len(graph.node_list) - 1:
            edge_matrix_str += ''
        else:
            edge_matrix_str += '<sep> '
    duty_cycle_order = {0.1:0, 0.3:1, 0.5:2, 0.7:3, 0.9:4}
    duty_one_hot = np.zeros((5))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_one_str = "Duty cycle: "
    for e in duty_one_hot:
        if int(e) == 0:
            duty_one_str += '<unselect> '
        elif int(e) == 1:
            duty_one_str += '<select> '
        else:
            raise NotImplementedError
    duty_one_str += '<sep> '
    # duty_one_str += ' '.join(str(int(e)) for e in duty_one_hot)
    # print('duty_one_hot', duty_one_hot)
    # print('edge_matrix_str', edge_matrix_str)
    # print('duty_one_str', duty_one_str)

    '''Version 1 duty cycle then graph'''
    circuit_str = duty_one_str + node_order_str + edge_matrix_str
    instruction = "Duty cycle options: 0.1, 0.3, 0.5, 0.7, 0.9. Voltage conversion ratio: {:.6f}. Efficiency: {:.6f}.".format(datum["vout"] / 100, datum["eff"])
    inputs = node_order_str
    outputs = duty_one_str + edge_matrix_str
    '''Version 2 graph then duty cycle'''
    # circuit_str = node_order_str + edge_matrix_str + duty_one_str
    # print('circuit_str', circuit_str)
    # input()
    d_dict['instruction'] = instruction
    d_dict['input'] = inputs
    d_dict['output'] = outputs
    # if len(graph.node_list) == 6:
    # print(circuit_str)
    return d_dict


def convert_raw_2_matrix_input_labels(raw_data):
    vouts = []
    effs = []
    for datum in tqdm(raw_data):
        # print(datum)
        vouts.append(datum['vout'])
        effs.append(datum['eff'])
    print(np.max(effs), np.min(effs))
    vouts = np.array(vouts)
    effs = np.array(effs)
    upper_threshold_power = np.percentile(vouts, 99.5)
    lower_threshold_power = np.percentile(vouts, 0.5)
    # upper_threshold_eff = np.percentile(effs, 99.5)
    upper_threshold_eff = 1.0
    # lower_threshold_eff = 0.0
    lower_threshold_eff = np.percentile(effs, 0.5)
    print('lower_threshold_eff', lower_threshold_eff)
    print('upper_threshold_eff', upper_threshold_eff)

    n_invalid = 0
    matrix_data = []
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            if datum["vout"] != -500:
                continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            if datum["eff"] != -1:
                continue
        if datum["vout"] == -500 and datum["eff"] == -1:
            n_invalid += 1
        d_dict = gen_adjmatrix_textdata_from_raw_input_labels(datum)
        # print(d_dict)
        # input()
        matrix_data.append(d_dict)
    print('n_invalid', n_invalid)
    print('len(matrix_data)', len(matrix_data))
    return matrix_data
    
    

def remove_isomorphism_circuit(raw_data, check_isomorphism=False):
    print('run remove_isomorphism_circuit in parsers/data_utils')
    voltage2data = {}
    idx = 0
    if type(raw_data) == dict:
        print('raw_data is dict')
        raw_data = list(raw_data.values())
    elif type(raw_data) == list:
        print('raw_data is list')
        pass
    else:
        raise NotImplementedError
    plt.figure()
    vouts = []
    effs = []
    n_valid_data = 0
    for datum in tqdm(raw_data):
        list_of_node = []
        # print(datum)
        for node in datum['list_of_node']:
            if type(node) == int:
                continue
            list_of_node.append(node)
        list_of_node.sort()
        node_str = ''
        for node in list_of_node:
            node_str += (' '+node)
        if (node_str, datum["vout"], datum["eff"]) in voltage2data.keys():
            voltage2data[(node_str, datum["vout"], datum["eff"])].append(datum)
        else:
            voltage2data[(node_str, datum["vout"], datum["eff"])] = [datum]
        
        # plt.scatter(datum["vout"], datum["eff"])
        vouts.append(datum["vout"])
        effs.append(datum["eff"])
        idx += 1
        # if datum["vout"] == -500:
        #     print("fuck")
        if datum["eff"] == -1:
            # print('eff = -1, vout = ', datum["vout"])
            # T = nx.Graph()
            # T.add_nodes_from(datum["list_of_node"])
            # T.add_edges_from(datum["list_of_edge"])
            # plt.figure()
            # nx.draw(T, node_color='white', with_labels=True)
            # plt.savefig('plot/component3_invalid.png', dpi=500)
            # plt.close()
            # input()
            pass
        else:
            n_valid_data += 1
    # print(min(effs))
    # input()
    # plt.xlabel('vout')
    # plt.ylabel('eff')
    # plt.savefig('plot/vout_eff7.png', dpi=500)
    # plt.close()
    # vouts = np.array(vouts)
    # upper_threshold_power = np.percentile(vouts, 99.5)
    # lower_threshold_power = np.percentile(vouts, 0.5)
    # mask = np.where(vouts == -500)[0]
    # vouts_invalid = np.array(vouts)[mask]
    # effs_invalid = np.array(effs)[mask]
    # # mask = np.logical_and(vouts >= -500, vouts <= -500)
    # # print(mask)
    # mask = np.logical_and(vouts > lower_threshold_power, vouts < upper_threshold_power)
    # vouts = np.array(vouts)[mask]
    # effs = np.array(effs)[mask]
    # upper_threshold_power = np.percentile(effs, 99.5)
    # lower_threshold_power = np.percentile(effs, 0.5)
    # mask = np.logical_and(effs > lower_threshold_power, effs < upper_threshold_power)
    # vouts = np.array(vouts)[mask]
    # effs = np.array(effs)[mask]
    # vouts = np.concatenate([vouts, vouts_invalid])
    # effs = np.concatenate([effs, effs_invalid])
    # print('min vout', min(vouts))
    # print('min eff', min(effs), lower_threshold_power)
    # # vouts = np.clip(vouts, lower_threshold_power, upper_threshold_power)/100
    # # plt.hist(vouts, bins=50)
    # # plt.savefig('plot/vout_hist7.png', dpi=300)
    # # plt.close()
    # # plt.hist(effs, bins=50)
    # # plt.savefig('plot/eff_hist7.png', dpi=300)
    # # plt.close()
    # # plt.hist2d(vouts, effs, bins=50, cmap='Blues')
    # # plt.xlabel('vout')
    # # plt.ylabel('eff')
    # # plt.savefig('plot/vout_eff_hist7.png', dpi=300)
    # # plt.close()
    # print('voltage2data', len(voltage2data))
    # print('raw_data', len(raw_data))
    # print('n_valid_data', n_valid_data)
    # input()
    new_raw_data = []
    n_invalid_data = 0
    for name, datums in tqdm(voltage2data.items()):
        if len(datums) == 1:
            new_raw_data.append(datums[0])
            continue
        idx = 0
        graphs = []
        for datum in datums:
            # path = 'sim_check.cki'
            # instruction, input_ids, output = gen_textdata_from_raw(datum)
            # print(datum)
            # result = sim_generation_output(path, output)
            # print(result)
            # input()
            topo_file = os.path.join('plot', 'figure{}.png'.format(idx))
            T = nx.Graph()
            for node in datum["list_of_node"]:
                if type(node) == int:
                    T.add_node(node, type='connection')
                elif node == 'VIN' or node == 'VOUT' or node == 'GND':
                    T.add_node(node, type=node)
                else:
                    T.add_node(node, type=node[:len(node)-1])
            T.add_edges_from(datum["list_of_edge"])
            graphs.append(T)
            # plt.figure()
            # nx.draw(T, with_labels=True)
            # plt.savefig(topo_file)
            # plt.close()
            idx += 1
        # print('len of graphs', len(graphs))
        graph_type_dict = {}
        distinct_graph_set = set()
        type_idx = 0
        if check_isomorphism:
            for idx1 in range(len(graphs)):
                for idx2 in range(idx1, len(graphs)):
                    if idx1 == idx2:
                        continue
                    assert(nx.vf2pp_is_isomorphic(graphs[idx1], graphs[idx2], node_label='type') == False)
        else:
            for idx1 in range(len(graphs)):
                if idx1 not in graph_type_dict.keys():
                    distinct_graph_set.add(idx1)
                    for idx2 in range(idx1, len(graphs)):
                        if idx1 == idx2:
                            continue
                        if idx2 in graph_type_dict.keys():
                            continue
                        # use for double-check the pruned dataset
                        # assert(nx.vf2pp_is_isomorphic(graphs[idx1], graphs[idx2], node_label='type') == False)
                        if nx.vf2pp_is_isomorphic(graphs[idx1], graphs[idx2], node_label='type'):
                            graph_type_dict[idx2] = idx1
                            # distinct_graph_set.add(idx1)
                            # if datums[idx1]['eff'] == -1:
                            #     print('graph {} and {} are isomorphic'.format(idx1, idx2))
                            #     plt.figure()
                            #     nx.draw(graphs[idx1], with_labels=True)
                            #     plt.savefig('plot/component3_invalid_iso1.png')
                            #     plt.close()
                            #     plt.figure()
                            #     nx.draw(graphs[idx2], with_labels=True)
                            #     plt.savefig('plot/component3_invalid_iso2.png')
                            #     plt.close()
                            #     input()
            # print(graph_type_dict)
            # print(distinct_graph_set)
            for dis_idx in distinct_graph_set:
                if datums[dis_idx]['eff'] == -1:
                    n_invalid_data += 1
                new_raw_data.append(datums[dis_idx])
    print('n_invalid_data', n_invalid_data)
    print('number of new_raw_data', len(new_raw_data))
    return new_raw_data

def extract_essensial_remove_redundant_circuit(raw_data):
    # raw_data = list(raw_data.values())
    def convert_graph_to_datum(T, datum):
        # print(T.nodes)
        # print(T.edges)
        datum_new = datum
        datum_new["list_of_node"] = list(T.nodes)
        datum_new["list_of_edge"] = [[a,b] for (a, b) in T.edges]
        # print(datum_new["list_of_node"], datum_new["list_of_edge"])
        return datum_new
    sub_graph_str_set = set()
    num = 0
    new_graphs = []
    datums_new = []
    for datum in tqdm(raw_data):
        graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
        T = nx.Graph()
        T.add_nodes_from(datum["list_of_node"])
        T.add_edges_from(datum["list_of_edge"])
        # if nx.is_tree(T):
        paths = graph.find_path_Vin_Vout()
        paths_to_GND = nx.all_simple_paths(T, source='GND', target='VIN')
        paths_to_VOUT = nx.all_simple_paths(T, source='VOUT', target='VIN')
        node_set = set()
        for path in paths_to_VOUT:
            for node in path:
                node_set.add(node)
        # print(node_set)
        T_copy = copy.deepcopy(T)
        is_removed = False
        moved_node = []
        for node in T.nodes:
            if node not in node_set:
                is_removed = True
                # print(node, 'is removed.')
                moved_node.append(node)
                T_copy.remove_node(node)
        if is_removed and not (len(moved_node) == 1 and moved_node[0] == 'GND'):
            assert(len(moved_node) != 1 or moved_node[0] != 'GND')
            total_path_str = ""
            for i, path in enumerate(paths):
                path_str = graph.encode_path_as_string(path)
                total_path_str += path_str
            # print(total_path_str)
            if total_path_str not in sub_graph_str_set:
                sub_graph_str_set.add(total_path_str)
                new_graphs.append(T_copy)
                # print('datum', datum["list_of_node"], datum["list_of_edge"])
                # print('original', datum)
                datum_new = convert_graph_to_datum(T_copy, datum)
                datums_new.append(datum_new)
                # result, output = sim_cir(datum_new)
                # print(output, '\n', result)
                # num+=1
                # topo_file = os.path.join('plot', 'tree.png')
                # plt.figure()
                # nx.draw(T_copy, node_color='white', with_labels=True)
                # plt.savefig(topo_file, dpi=500)
                # plt.close()
                # topo_file = os.path.join('plot', 'tree_org.png')
                # plt.figure()
                # nx.draw(T, node_color='white', with_labels=True)
                # plt.savefig(topo_file, dpi=500)
                # plt.close()
                # input()
            # input()
    # print(sub_graph_str_set)
    print(len(sub_graph_str_set))
    return datums_new

def check_generated_isomorphism(dset, dset_val, data_generated, scalar_eff_labels, scalar_vout_labels):
    duty_cycle_options = [0.1, 0.3, 0.5, 0.7, 0.9]
    node_tokens = set()
    type_str = ['Sa', 'Sb', 'C', 'L']
    for device in type_str:
        for i in range(5):
            device_str = device + str(i)
            node_tokens.add(device_str)
    node_tokens.add('IN')
    node_tokens.add('OUT')
    node_tokens.add('0')
    trn_graphs = []
    trn_duty_cycles = []
    trn_cir_strs = []
    trn_effs = []
    trn_vouts = []
    for datum in dset:
        # print(datum)
        circuit_str = datum['circuit_str']
        netlist, duty_cycle = read_masked_LLM_output(datum['circuit_str'])
        # netlist, duty_cycle = read_LLM_ouput(datum['circuit_str'])
        # netlist, duty_cycle = read_LLM_output_shrink_canonical(datum['output'])
        
        graph = convert_netlist_2_graph(node_tokens, netlist)
        trn_graphs.append(graph)
        trn_duty_cycles.append(duty_cycle)
        trn_cir_strs.append(circuit_str)
        trn_effs.append(datum['eff'])
        trn_vouts.append(datum['vout'])
    trn_effs = np.array(trn_effs)
    trn_vouts = np.array(trn_vouts)
    # xy = np.vstack([trn_effs, trn_vouts])
    # z = gaussian_kde(xy)(xy)
    # plt.scatter(trn_effs, trn_vouts, c='black', s=5, label='train')
    val_effs = []
    val_vouts = []
    val_cir_strs = []
    for datum in dset_val:
        if datum['eff'] == -1 or type(datum['vout']) == int:
            continue
        val_effs.append(datum['eff'])
        val_vouts.append(datum['vout'])
        val_cir_strs.append(datum['circuit_str'])
        # netlist, duty_cycle = read_masked_LLM_output(datum['circuit_str'])
        # graph = convert_netlist_2_graph(node_tokens, netlist)
    # print('val len: ', len(val_effs))
    total_d_num = 0
    new_cir_num = 0
    new_cir_eff_logits = []
    new_cir_eff_labels = []
    new_cir_vout_logits = []
    new_cir_vout_labels = []

    data_generated_valid = []
    for idx, datum in enumerate(tqdm(data_generated)):
        if datum["result"]['result_valid'] == False:
            continue
        data_generated_valid.append(datum)
    print('valid len: ', len(data_generated_valid))
    print('len of scalar_eff_labels: ', len(scalar_eff_labels))

    for idx, datum in enumerate(tqdm(data_generated_valid)):
        total_d_num += 1
        inputs = datum['input']
        output = datum['output']
        # print(inputs, output)
        # input()
        # st_token_index = inputs.find('<extra_id_0>')
        # inputs = inputs[st_token_index-11:]
        circuit_str = combine_masked_input_output(inputs, output)
        netlist, duty_cycle = read_masked_LLM_output(circuit_str)
        # netlist, duty_cycle = read_LLM_ouput(output)
        # netlist, duty_cycle = read_LLM_output_shrink_canonical(output)
        # netlist, duty_cycle = read_LLM_output_shrink_canonical_dutycycle(output)
        # netlist, duty_cycle = read_transformer_output_shrink_canonical(output)
        # netlist, duty_cycle = read_transformer_output_mask(inputs, output)
        graph = convert_netlist_2_graph(node_tokens, netlist)
        brand_new = True
        # for i, trn_graph in enumerate(trn_graphs):
        #     if duty_cycle == trn_duty_cycles[i] and nx.vf2pp_is_isomorphic(trn_graph, graph, node_label='type'):
        #         brand_new = False
        #         # plt.scatter(datum['result']['efficiency'], datum['result']['Vout']/100, c='blue', s=5, label='val_label')
        #         # plt.scatter(val_effs[idx], val_vouts[idx], c='orange', s=5, label='gen_from_train')
        #         # plt.scatter(val_effs[idx], datum['result']['efficiency'], c='blue', s=5)
        #         # plt.scatter(val_vouts[idx], datum['result']['Vout']/100, c='blue', s=5)
        #         break
                # print('trn: ', trn_cir_strs[i])
                # print('trn_eff: ', trn_effs[i])
                # print('trn_vout: ', trn_vouts[i])
                # print('gen: ', circuit_str)
                # print('gen_eff: ', datum['result']['efficiency'])
                # print('gen_vout: ', datum['result']['Vout'])
                # print('val require eff: ', val_effs[idx])
                # print('val require vout: ', val_vouts[idx])
                # print('isomorphic!!!!')
                # input()
        # if brand_new:
            # for dc in duty_cycle_options:
            #     path = "sim_check.cki"
            #     result = sim_netlist_duty_cycle(path, netlist, dc)
            #     print('dc: ', dc)
            #     print('result: ', result)
            # plt.scatter(val_effs[idx], val_vouts[idx], c='red', s=5, label='gen_new')
            # plt.scatter(val_effs[idx], datum['result']['efficiency'], c='red', s=10)
            # plt.scatter(val_vouts[idx], datum['result']['Vout']/100, c='red', s=10)
        if datum['result']['efficiency'] > 0.9:
            if datum['result']['Vout']/100 > 0.4 and datum['result']['Vout']/100 < 0.6 or datum['result']['Vout']/100 > 1.2 and datum['result']['Vout']/100 < 3.0:
                print('gen_eff: ', datum['result']['efficiency'])
                print('gen_vout: ', datum['result']['Vout']/100)
                print(netlist, duty_cycle)
                # for i, trn_graph in enumerate(trn_graphs):
                #     if duty_cycle == trn_duty_cycles[i] and nx.vf2pp_is_isomorphic(trn_graph, graph, node_label='type'):
                #         brand_new = False
                #         break
                # if brand_new:
                input()
        if brand_new:
            new_cir_eff_logits.append(datum['result']['efficiency'])
            new_cir_eff_labels.append(scalar_eff_labels[idx])
            new_cir_vout_logits.append(datum['result']['Vout']/100)
            new_cir_vout_labels.append(scalar_vout_labels[idx])

            new_cir_num += 1
            print('gen   eff and vout: ', datum['result']['efficiency'], ', ', datum['result']['Vout']/100)
            print('label eff and vout: ', scalar_eff_labels[idx], ', ', scalar_vout_labels[idx])
            print('brand new circuit!!!! with number ', new_cir_num)
        # print('gen duty cycle: ', duty_cycle)
        # print('gen_eff: ', datum['result']['efficiency'])
        # print('gen_vout: ', datum['result']['Vout']/100)
        # print('val require eff: ', scalar_eff_labels[idx])
        # print('val require vout: ', scalar_vout_labels[idx])
            # print('gen: \n', circuit_str)
            # print('val circuit: \n', val_cir_strs[idx])
            # input()
    print('total_d_num: ', total_d_num)
    print('new_cir_num: ', new_cir_num)
    print('new_cir_num/total_d_num: ', new_cir_num/total_d_num)
    # plt.xlabel('Efficiency')
    # plt.ylabel('Voltage conversion ratio')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.plot(val_effs, val_effs, linewidth=0.5)
    # plt.xlabel('Efficiency labels')
    # plt.ylabel('Efficiency predictions')
    # plt.xlim([-3, 2.5])
    # plt.ylim([-3, 2.5])
    # plt.plot(val_vouts, val_vouts, linewidth=0.5)
    # plt.xlabel('Voltage conversion ratio labels')
    # plt.ylabel('Voltage conversion ratio predictions')
    # # plt.legend()
    # # plt.savefig("plot/eff_masked_new_gen_connect_full.png", dpi=300)
    # plt.savefig("plot/vout_masked_new_gen_connect_full.png", dpi=300)
    return new_cir_eff_logits, new_cir_eff_labels, new_cir_vout_logits, new_cir_vout_labels

def random_generate_graph_6():
    data_num = 1000
    node_order = {'VIN':0, 'VOUT':1, 'GND':2}
    type_str = ['Sa', 'Sb', 'C', 'L']
    idx = 3
    for device in type_str:
        for i in range(8):
            device_str = device + str(i)
            node_order[device_str] = idx
            idx += 1
    # node_list = []
    # for node in self.node_list:
    #     if type(node) == int:
    #         continue
    #     node_list.append(node)
    # node_list.sort(key=lambda val: node_order[val])
    duty_cycle_options = [0.1, 0.3, 0.5, 0.7, 0.9]
    for _ in range(data_num):
        node_list = ['VIN', 'VOUT', 'GND']
        type_str = ['Sa', 'Sb', 'C', 'L']
        node_type_id = {'Sa':0, 'Sb':1, 'C':2, 'L':3}
        node_type_num = [0, 0, 0, 0]
        for j in range(6):
            node_type = np.random.choice(type_str)
            print('node_type: ', node_type)
            node_list.append(node_type + str(node_type_num[node_type_id[node_type]]))
            node_type_num[node_type_id[node_type]] += 1
        print('node_type_num: ', node_type_num)
        node_list.sort(key=lambda val: node_order[val])
        print(node_list)
        input()
        adj_matrix = np.zeros((len(node_list), len(node_list)))
        # print('node_list', self.node_list)
        node2id = {}
        for i, node in enumerate(node_list):
            node2id[node] = i
        for i, node in enumerate(node_list):
            ## help me to random generate the graph structure
            if i < 3:
                continue





if __name__ == '__main__':
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
    data_path = os.path.join(prefix, "data1.json")
    print('data_path', data_path)
    raw_data = json.load(open(data_path, 'r'))
    for name, datum in tqdm(raw_data.items()):
        instruction, input_ids, output = gen_textdata_from_raw(datum)
        print(output)
        gen_adjmatrix_textdata_from_raw(datum)
        input()
