import argparse
import sys
import os
dir_path = os.getcwd()
sys.path.append(dir_path)

import json

import numpy as np
from tqdm import tqdm

from util import *

from topo_data_util.topo_analysis.topoGraph import TopoGraph
from topo_data_util.topo_utils.plot import plot_hist
from utils.yaml_parser import load_and_apply_yaml_config

def transform_raw_2_naive_form(datum):
    """
    [LaMAGIC, ICML'24] Utility function
    Transform raw data to Naive formulation for edge generation task
    """
    
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    phase_one_switches = []
    phase_two_switches = []
    capacitances = []
    inductors = []
    ports = []
    input = ""
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
    
    input += gen_string(phase_one_switches, "phase-one switch")
    input += gen_string(phase_two_switches, "phase-two switch")
    input += gen_string(inductors, "inductor")
    input += gen_string(capacitances, "capacitance")
    input += "a circuit input VIN, a circuit output VOUT, a ground GND. "
    input += "The duty cycle has five options (0.1, 0.3, 0.5, 0.7, 0.9). "
    vout = datum["vout"] / 100
    input += "The target power conversion ratio is {:.6f}, and the efficiency is {:.6f}".format(vout, datum["eff"])
    
    instruction = 'Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve the target power conversion ratio and efficiency.'
    
    return instruction, input, output

def transform_raw_2_naive_form_topology(datum):
    """
    [LaMAGIC, ICML'24] Utility function
    Transform raw data to Naive formulation for topology generation task
    """
    
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    phase_one_switches = []
    phase_two_switches = []
    capacitances = []
    inductors = []
    ports = []
    input = ""
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

    vout = datum["vout"] / 100
    input += "The target power conversion ratio is {:.6f}, and the efficiency is {:.6f}".format(vout, datum["eff"])
    
    instruction = 'Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve the target power conversion ratio and efficiency.'
    
    return instruction, input, output

def gen_textdata_from_raw_shrink_canonical(datum):
    """
    [LaMAGIC, ICML'24] Utility function
    Transform raw data to Canonical formulation
    """
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.sort_hyper_edges()
    instruction = "Duty cycle options: 0.1, 0.3, 0.5, 0.7, 0.9. Voltage conversion ratio: {:.6f}. Efficiency: {:.6f}.".format(datum["vout"] / 100, datum["eff"])
    inputs = "Vertex order: "
    for i, node in enumerate(graph.node_list):
        # if node == 'VIN' or node == 'VOUT' or node == 'GND':
        inputs += node
        # else:
        #     inputs += node[:-1]
        
        if i != len(graph.node_list) - 1:
            inputs += ' '
    inputs += '.'
    output = "Connections: "
    for i, edge in enumerate(graph.hyper_edge_list):
        # edge.reduce_number()
        output += str(edge)
        if i != len(graph.hyper_edge_list) - 1:
            output += ', '
    output += "\nThe duty cycle is set to {}.".format(datum["duty_cycle"])
    return instruction, inputs, output

def gen_textdata_from_raw_shrink_canonical_dutycycle(datum):
    """
    [LaMAGIC, ICML'24] Utility function
    Transform raw data to Canonical formulation + one-hot encoding
    """
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.sort_hyper_edges()
    instruction = "Duty cycle options: 0.1, 0.3, 0.5, 0.7, 0.9. Voltage conversion ratio: {:.6f}. Efficiency: {:.6f}.".format(datum["vout"] / 100, datum["eff"])
    inputs = "Vertex order: "
    for i, node in enumerate(graph.node_list):
        # if node == 'VIN' or node == 'VOUT' or node == 'GND':
        inputs += node
        # else:
        #     inputs += node[:-1]
        
        if i != len(graph.node_list) - 1:
            inputs += ' '
    # inputs += '.'
    output = "Connections: "
    for i, edge in enumerate(graph.hyper_edge_list):
        # edge.reduce_number()
        output += str(edge)
        if i != len(graph.hyper_edge_list) - 1:
            output += ', '
    # output += '.'
    duty_cycle_order = {0.1:0, 0.3:1, 0.5:2, 0.7:3, 0.9:4}
    duty_one_hot = np.zeros((5))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    output += "\nDuty cycle: "
    for e in duty_one_hot:
        if int(e) == 0:
            output += '<unselect>'
        elif int(e) == 1:
            output += '<select>'
        else:
            raise NotImplementedError
    # output += '.'
    return instruction, inputs, output

def gen_adjmatrix_textdata_from_raw(datum):
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
        d_dict = gen_adjmatrix_textdata_from_raw(datum)
        matrix_data.append(d_dict)
    print('n_invalid', n_invalid)
    print('len(matrix_data)', len(matrix_data))
    return matrix_data

def parse_transform_raw_2_naive(data_path, output_path):
    """
    [LaMAGIC, ICML'24] Utility function
    Transform raw data to Naive formulation for edge generation task
    """
    raw_data = json.load(open(data_path, 'r'))
    vouts = []
    effs = []
    for datum in tqdm(raw_data):
        vouts.append(datum['vout'])
        effs.append(datum['eff'])
    vouts = np.array(vouts)
    effs = np.array(effs)
    upper_threshold_power = np.percentile(vouts, 99.5)
    lower_threshold_power = np.percentile(vouts, 0.5)
    upper_threshold_eff = 1.0
    lower_threshold_eff = np.percentile(effs, 0.5)

    n_invalid = 0
    data_text = []
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            if datum["vout"] != -500:
                continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            if datum["eff"] != -1:
                continue
        if datum["vout"] == -500 and datum["eff"] == -1:
            n_invalid += 1
        instruction, inputs, output = transform_raw_2_naive_form(datum)
        d_dict = {}
        d_dict["instruction"] = instruction
        d_dict["input"] = inputs
        d_dict["output"] = output
        data_text.append(d_dict)
    print("### Collect totally {} of data".format(len(data_text)))
    with open(output_path, 'w') as f:
        json.dump(data_text, f)
    

def parse_transform_raw_2_naive_form_topology(data_path, output_path):
    raw_data = json.load(open(data_path, 'r'))
    vouts = []
    effs = []
    for datum in tqdm(raw_data):
        vouts.append(datum['vout'])
        effs.append(datum['eff'])
    vouts = np.array(vouts)
    effs = np.array(effs)
    upper_threshold_power = np.percentile(vouts, 99.5)
    lower_threshold_power = np.percentile(vouts, 0.5)
    upper_threshold_eff = 1.0
    lower_threshold_eff = np.percentile(effs, 0.5)

    n_invalid = 0
    data_text = []
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            if datum["vout"] != -500:
                continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            if datum["eff"] != -1:
                continue
        if datum["vout"] == -500 and datum["eff"] == -1:
            n_invalid += 1
        instruction, inputs, output = transform_raw_2_naive_form_topology(datum)
        d_dict = {}
        d_dict["instruction"] = instruction
        d_dict["input"] = inputs
        d_dict["output"] = output
        data_text.append(d_dict)
    print("### Collect totally {} of data".format(len(data_text)))
    with open(output_path, 'w') as f:
        json.dump(data_text, f)

def parse_json_data_shrink_canonical(data_path, output_path):
    """
    [LaMAGIC, ICML'24]
    Transform raw data to Canonical formulation
    """
    raw_data = json.load(open(data_path, 'r'))
    vouts = []
    effs = []
    for datum in tqdm(raw_data):
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
    # print('lower_threshold_eff', lower_threshold_eff)
    # print('upper_threshold_eff', upper_threshold_eff)

    n_invalid = 0
    data_text = []
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            if datum["vout"] != -500:
                continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            if datum["eff"] != -1:
                continue
        if datum["vout"] == -500 and datum["eff"] == -1:
            n_invalid += 1
        instruction, inputs, output = gen_textdata_from_raw_shrink_canonical(datum)
        # print(instruction)
        # print(inputs)
        # print(output)
        # input()
        d_dict = {}
        d_dict["instruction"] = instruction
        d_dict["input"] = inputs
        d_dict["output"] = output
        data_text.append(d_dict)
    print("### Collect totally {} of data".format(len(data_text)))
    with open(output_path, 'w') as f:
        json.dump(data_text, f)

def parse_json_data_shrink_canonical_dutycycle(data_path, output_path):
    """
    [LaMAGIC, ICML'24]
    Transform raw data to Canonical formulation + one-hot encoding
    """
    raw_data = json.load(open(data_path, 'r'))
    vouts = []
    effs = []
    for datum in tqdm(raw_data):
        vouts.append(datum['vout'])
        effs.append(datum['eff'])
    print(np.max(effs), np.min(effs))
    vouts = np.array(vouts)
    effs = np.array(effs)
    upper_threshold_power = np.percentile(vouts, 99.5)
    lower_threshold_power = np.percentile(vouts, 0.5)
    upper_threshold_eff = 1.0
    lower_threshold_eff = np.percentile(effs, 0.5)

    n_invalid = 0
    data_text = []
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            if datum["vout"] != -500:
                continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            if datum["eff"] != -1:
                continue
        if datum["vout"] == -500 and datum["eff"] == -1:
            n_invalid += 1
        instruction, inputs, output = gen_textdata_from_raw_shrink_canonical_dutycycle(datum)
        # print(instruction)
        # print(inputs)
        # print(output)
        # input()
        d_dict = {}
        d_dict["instruction"] = instruction
        d_dict["input"] = inputs
        d_dict["output"] = output
        data_text.append(d_dict)
    print("### Collect totally {} of data".format(len(data_text)))
    with open(output_path, 'w') as f:
        json.dump(data_text, f)

def parse_json_data_convert_raw_2_matrix(data_path, output_path):
    """
    [LaMAGIC, ICML'24]
    Transform raw data to Adjaency matrix formulation
    """
    print("RUNNING convert_raw_2_matrix_main")
    raw_data = json.load(open(data_path, 'r'))
    matrix_data = convert_raw_2_matrix(raw_data)
    with open(output_path, 'w') as f:
        json.dump(matrix_data, f)

if __name__ == '__main__':
    

    """
    Use 1. parse_transform_raw_2_naive(data_path, output_path) for Naive formulation for edge generation task
    Use 2. parse_transform_raw_2_naive_form_topology(data_path, output_path) for Naive formulation for topology generation task
    Use 3. parse_json_data_shrink_canonical(data_path, output_path) for Canonical formulation
    Use 4. parse_json_data_shrink_canonical_dutycycle(data_path, output_path) for Canonical formulation + duty cycle one-hot encoding
    Use 5. parse_json_data_convert_raw_2_matrix(data_path, output_path) for two adjacency matrix formulations FM and PM

    Example: for Naive formulation for topology generation task
    """
    # the raw data path for 345-component circuits
    prefix = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/text_dataset/masked"
    data_path = prefix + "dataset_all_345_regenerate_prune_isomophic_new.json"
    # the output path for the text data after transformation
    output_path = os.path.join('/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523', 'dataset_all_345_regenerate_prune_isomophic_topology.json')
    parse_json_data_convert_raw_2_matrix(data_path=data_path, output_path=output_path)

    # raw data path for 6-component circuits
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/new_dataset/6_component_raw/regenerate_30000'
    data_path = os.path.join(prefix, 'dataset_6_regenerate_30000_remove_isomophism.json')
    output_path = os.path.join("/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/text_dataset/masked", 'dataset_6_regenerate.json')
    parse_json_data_convert_raw_2_matrix(data_path=data_path, output_path=output_path)
    exit()