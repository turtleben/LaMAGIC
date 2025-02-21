import argparse
import sys
import os
import copy
dir_path = os.getcwd()
sys.path.append(dir_path)

import itertools
import json
from matplotlib import pyplot as plt

import random
import numpy as np
# import wandb
import torch
from tqdm import tqdm

from transformer_args import get_transformer_args
from scipy.stats import gaussian_kde
from util import *

# sys.path.append(os.path.join(sys.path[0], '../topo_data_util/'))
# from train import main as train_fn
from topo_data_util.topo_analysis.topoGraph import TopoGraph
from GetReward import calculate_reward
from topo_data_util.topo_utils.plot import plot_hist
from utils.yaml_parser import load_and_apply_yaml_config
from parsers.simulation import sim_generation_output

def gen_textdata_from_raw(datum):
    
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
    
    input += gen_string(phase_one_switches, "phase-one switch")
    input += gen_string(phase_two_switches, "phase-two switch")
    input += gen_string(inductors, "inductor")
    input += gen_string(capacitances, "capacitance")
    input += "a circuit input VIN, a circuit output VOUT, a ground GND. "
    input += "The duty cycle has five options (0.1, 0.3, 0.5, 0.7, 0.9). "
    # for i, port in enumerate(ports):
    #     if i == len(port) - 1:
    #         input += " and"
    #     input += "{}, ".format(port)
    # instruction += "by setting duty cycle to {}, ".format(datum["duty_cycle"])
    vout = datum["vout"] / 100
    input += "The target power conversion ratio is {:.6f}, and the efficiency is {:.6f}".format(vout, datum["eff"])
    # torch.clamp(vout / 100., 0., 1.)
    
    # instruction += 'generate a circuit topology and select the duty cycle with options {{0.1, 0.3, 0.5, 0.7, 0.9}} to achieve a target power conversion ratio of {:.2f}'.format(vout)
    # instruction = 'Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve a target power conversion ratio of {:.2f}.'.format(vout)
    instruction = 'Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve the target power conversion ratio and efficiency.'
    
    return instruction, input, output



def gen_textdata_from_raw_shrink_canonical_dutycycle_first(datum):
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.sort_hyper_edges()
    instruction = "Duty cycle options: 0.1, 0.3, 0.5, 0.7, 0.9. Voltage conversion ratio: {:.6f}. Efficiency: {:.6f}.".format(datum["vout"] / 100, datum["eff"])
    inputs = ""
    for i, node in enumerate(graph.node_list):
        # if node == 'VIN' or node == 'VOUT' or node == 'GND':
        inputs += node
        # else:
        #     inputs += node[:-1]
        
        if i != len(graph.node_list) - 1:
            inputs += ' '
    inputs += ' <sep> '
    duty_cycle_order = {0.1:0, 0.2:1, 0.3:2, 0.4:3, 0.5:4, 0.6:5, 0.7:6, 0.8:7, 0.9:8}
    duty_one_hot = np.zeros((10))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_cycle_tokens = ['<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>']
    output = " "
    for i, e in enumerate(duty_one_hot):
        if int(e) == 0:
            continue
        elif int(e) == 1:
            output += (duty_cycle_tokens[i] + ' ')
            break
        else:
            raise NotImplementedError
    output += ' <sep> '
    output += " "
    for i, edge in enumerate(graph.hyper_edge_list):
        # edge.reduce_number()
        for j, node in enumerate(edge.node_list):
            output += (node + ' ')
            if j == len(edge.node_list) - 1 and i != len(graph.hyper_edge_list) - 1:
                output += ' , '
    output += ' <sep>'

    node_num = 0
    for node in graph.node_list:
        if type(node) == int:
            continue
        node_num += 1
    edge_num = len(graph.hyper_edge_list)
    hyperedge_node_num = 0
    for hyper_edge in graph.hyper_edge_list:
        hyperedge_node_num += len(hyper_edge.node_list)

    

    # original_output = "Connections: "
    # for i, edge in enumerate(graph.hyper_edge_list):
    #     # edge.reduce_number()
    #     original_output += str(edge)
    #     if i != len(graph.hyper_edge_list) - 1:
    #         original_output += ', '
    # print(original_output)
    # print(output)
    # input()
    
    
    # output += '.'
    return inputs, output, node_num, edge_num, hyperedge_node_num

def gen_textdata_from_raw_shrink_canonical_typeNidx_dutycycle_first(datum, common_word=False):
    common_word_dict = {'VIN': 'A', 'VOUT': 'B', 'GND': 'C', 'Sa': 'D', 'Sb': 'E', 'C': 'F', 'L': 'G'}
    
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.sort_hyper_edges()
    inputs = ""
    node_idx_map = {}
    idx = 0
    for i, node in enumerate(graph.node_list):
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            if not common_word:
                inputs += node
            else:
                inputs += common_word_dict[node]
        else:
            if node not in node_idx_map:
                node_idx_map[node] = idx
                idx += 1
            if not common_word:
                inputs += (node[:-1] + ' ' + str(node_idx_map[node]) + ' ')
            else:
                inputs += (common_word_dict[node[:-1]] + ' ' + str(node_idx_map[node]) + ' ')
        # if node == 'VIN' or node == 'VOUT' or node == 'GND':
        # inputs += node
        # else:
        #     inputs += node[:-1]
        
        if i != len(graph.node_list) - 1:
            inputs += ' '
    inputs += ' <sep> '
    duty_cycle_order = {0.1:0, 0.2:1, 0.3:2, 0.4:3, 0.5:4, 0.6:5, 0.7:6, 0.8:7, 0.9:8}
    duty_one_hot = np.zeros((10))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_cycle_tokens = ['<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>']
    output = " "
    for i, e in enumerate(duty_one_hot):
        if int(e) == 0:
            continue
        elif int(e) == 1:
            output += (duty_cycle_tokens[i] + ' ')
            break
        else:
            raise NotImplementedError
    output += ' <sep> '
    output += " "
    for i, edge in enumerate(graph.hyper_edge_list):
        # edge.reduce_number()
        for j, node in enumerate(edge.node_list):
            if node == 'VIN' or node == 'VOUT' or node == 'GND':
                if not common_word:
                    output += (node + ' ')
                else:
                    output += (common_word_dict[node] + ' ')
            else:
                if not common_word:
                    output += (node[:-1] + ' ' + str(node_idx_map[node]) + ' ')
                else:
                    output += (common_word_dict[node[:-1]] + ' ' + str(node_idx_map[node]) + ' ')
            
            if j == len(edge.node_list) - 1 and i != len(graph.hyper_edge_list) - 1:
                output += ' , '
    output += ' <sep>'

    node_num = 0
    for node in graph.node_list:
        if type(node) == int:
            continue
        node_num += 1
    edge_num = len(graph.hyper_edge_list)
    hyperedge_node_num = 0
    for hyper_edge in graph.hyper_edge_list:
        hyperedge_node_num += len(hyper_edge.node_list)

    return inputs, output, node_num, edge_num, hyperedge_node_num

def gen_textdata_from_raw_shrink_canonical_typeNidx_output_no_type_dutycycle_first(datum):
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.sort_hyper_edges()
    inputs = ""
    node_idx_map = {}
    idx = 0
    for i, node in enumerate(graph.node_list):
        if node not in node_idx_map:
            node_idx_map[node] = idx
            idx += 1
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            inputs += (node + ' ' + str(node_idx_map[node]) + ' ')
        else:
            inputs += (node[:-1] + ' ' + str(node_idx_map[node]) + ' ')
        # if node == 'VIN' or node == 'VOUT' or node == 'GND':
        # inputs += node
        # else:
        #     inputs += node[:-1]
        
        if i != len(graph.node_list) - 1:
            inputs += ' '
    inputs += ' <sep> '
    duty_cycle_order = {0.1:0, 0.2:1, 0.3:2, 0.4:3, 0.5:4, 0.6:5, 0.7:6, 0.8:7, 0.9:8}
    duty_one_hot = np.zeros((10))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_cycle_tokens = ['<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>']
    output = " "
    for i, e in enumerate(duty_one_hot):
        if int(e) == 0:
            continue
        elif int(e) == 1:
            output += (duty_cycle_tokens[i] + ' ')
            break
        else:
            raise NotImplementedError
    output += ' <sep> '
    output += " "
    for i, edge in enumerate(graph.hyper_edge_list):
        # edge.reduce_number()
        for j, node in enumerate(edge.node_list):
            output += (str(node_idx_map[node]) + ' ')
            
            if j == len(edge.node_list) - 1 and i != len(graph.hyper_edge_list) - 1:
                output += ' , '
    output += ' <sep>'
    node_num = 0
    for node in graph.node_list:
        if type(node) == int:
            continue
        node_num += 1
    edge_num = len(graph.hyper_edge_list)
    hyperedge_node_num = 0
    for hyper_edge in graph.hyper_edge_list:
        hyperedge_node_num += len(hyper_edge.node_list)

    return inputs, output, node_num, edge_num, hyperedge_node_num

def gen_volcabulary_canonical():
    
    node_tokens = ["<pad>", "</s>", "<unk>", '<sep>', ',',  '<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>', 'VIN', 'VOUT', 'GND']
    type_str = ['Sa', 'Sb', 'C', 'L']
    for device in type_str:
        for i in range(8):
            device_str = device + str(i)
            node_tokens.append(device_str)
    vocab = {}
    for i, token in enumerate(node_tokens):
        vocab[token] = i 

    with open("analog_LLM/configs/pure_transformer/dict/canonical_duty10first.json", 'w') as vocab_file:
        json.dump(vocab, vocab_file)

    return vocab

def gen_volcabulary_canonical_typeNidx():
    node_tokens = ["<pad>", "</s>", "<unk>", '<sep>', ',',  '<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>', 'VIN', 'VOUT', 'GND', 'Sa', 'Sb', 'C', 'L']
    for i in range(13):
        node_tokens.append(str(i))
    vocab = {}
    for i, token in enumerate(node_tokens):
        vocab[token] = i
    
    with open("analog_LLM/configs/pure_transformer/dict/canonical_typeNidx_duty10first.json", 'w') as vocab_file:
        json.dump(vocab, vocab_file)

def gen_volcabulary_matrix():
    
    node_tokens = ["<pad>", "</s>", "<unk>", '<sep>', ',',  '<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>', 'VIN', 'VOUT', 'GND', 'Sa', 'Sb', 'C', 'L', '<no_edge>', '<edge_1>', '<edge_2>', '<both_edges>']
    vocab = {}
    for i, token in enumerate(node_tokens):
        vocab[token] = i 

    with open("analog_LLM/configs/pure_transformer/dict/matrix_duty10first.json", 'w') as vocab_file:
        json.dump(vocab, vocab_file)

    return vocab
    
def parse_json_data_shrink_canonical_dutycycle_first(data_paths, output_path, target_vout=50, select_cond='none', use_log=False, six_comp=False, typeNidx=False, output_no_type=False, common_word=False):
    data_texts = []
    total_node_num = 0
    total_edge_num = 0
    total_hyperedge_node_num = 0
    for data_path in data_paths:
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
        if six_comp:
            lower_threshold_eff = 0.0
        else:
            lower_threshold_eff = np.percentile(effs, 0.5)
        # print('lower_threshold_eff', lower_threshold_eff)
        # print('upper_threshold_eff', upper_threshold_eff)

        n_invalid = 0
        data_text = []
        vouts = []
        effs = []
        for datum in tqdm(raw_data):
            if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
                if datum["vout"] != -500:
                    continue
            if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
                if datum["eff"] != -1:
                    continue
            if datum["vout"] == -500 and datum["eff"] == -1:
                n_invalid += 1
            if typeNidx == False:
                inputs, output, node_num, edge_num, hyperedge_node_num = gen_textdata_from_raw_shrink_canonical_dutycycle_first(datum)
            else:
                if output_no_type:
                    inputs, output, node_num, edge_num, hyperedge_node_num = gen_textdata_from_raw_shrink_canonical_typeNidx_output_no_type_dutycycle_first(datum)
                    # print(inputs)
                    # print(output)
                else:
                    inputs, output, node_num, edge_num, hyperedge_node_num = gen_textdata_from_raw_shrink_canonical_typeNidx_dutycycle_first(datum, common_word=common_word)
            total_node_num += node_num
            total_edge_num += edge_num
            total_hyperedge_node_num += hyperedge_node_num
            # print(inputs)
            # print(output)
            # # print(datum['list_of_edge'])
            # input()
            d_dict = {}
            # d_dict["instruction"] = instruction
            d_dict["d_cycle_option"] = [0.1, 0.3, 0.5, 0.7, 0.9]
            d_dict["vout"] = datum["vout"] / 100
            d_dict["eff"] = datum["eff"]
            vouts.append(datum["vout"] / 100)
            effs.append(datum["eff"])
            d_dict["input"] = inputs
            d_dict["output"] = output
            data_text.append(d_dict)
        data_texts.append(data_text)
        print("### Collect totally {} of data".format(len(data_text)))
    data_text = list(itertools.chain(*data_texts))
    print("### Collect totally {} of data".format(len(data_text)))
    print("average node num", float(total_node_num) / len(data_text))
    print("average edge num", float(total_edge_num) / len(data_text))
    print("average hyperedge node num", float(total_hyperedge_node_num) / len(data_text))
    # xy = np.vstack([vouts, effs])
    # z = gaussian_kde(xy)(xy)
    # plt.scatter(vouts, effs, s=8, c=z)
    # plt.xlim([-1, 1.4])
    # plt.ylim([0, 1])
    # plt.xlabel('vouts')
    # plt.ylabel('effs')
    # plt.savefig('vout_eff.png', dpi=200)
    # plt.close()
    input('press enter to save')
    with open(output_path, 'w') as f:
        json.dump(data_text, f)


def gen_textdata_from_raw_matrix_dutycycle_first(datum):
    graph = TopoGraph(node_list=datum['list_of_node'], edge_list=datum['list_of_edge'])
    graph.hyper_edges2adj_matrix()
    # print(datum)
    # print(graph.adj_matrix)
    d_dict = {}
    d_dict['vout'] = datum["vout"] / 100
    d_dict['eff'] = datum["eff"]

    inputs = ""
    for i, node in enumerate(graph.node_list):
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            inputs += node
        else:
            inputs += node[:-1]
        
        if i != len(graph.node_list) - 1:
            inputs += ' '
    inputs += ' <sep> '

    duty_cycle_order = {0.1:0, 0.2:1, 0.3:2, 0.4:3, 0.5:4, 0.6:5, 0.7:6, 0.8:7, 0.9:8}
    duty_one_hot = np.zeros((10))
    duty_one_hot[duty_cycle_order[datum["duty_cycle"]]] = 1
    duty_cycle_tokens = ['<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>']
    output = " "
    for i, e in enumerate(duty_one_hot):
        if int(e) == 0:
            continue
        elif int(e) == 1:
            output += (duty_cycle_tokens[i] + ' ')
            break
        else:
            raise NotImplementedError
    output += ' <sep> '
    output += " "
    # print('node_order_str', node_order_str)
    edge_matrix_str = ""
    has_both_edges = False
    for i, node in enumerate(graph.node_list):
        occur_1 = False
        swap_1_2 = False
        if node == 'VIN' or node == 'VOUT' or node == 'GND':
            edge_matrix_str += (node + ' ')
        else:
            edge_matrix_str += (node[:-1] + ' ')
        for j, e in enumerate(graph.adj_matrix[i]):
            # if j <= i:
            #     continue
            if int(e) == 0:
                edge_matrix_str += '<no_edge> '
            elif int(e) == 1 or int(e) == 2:
                if occur_1 == False and int(e) == 2:
                    swap_1_2 = True
                    occur_1 = True
                elif occur_1 == False and int(e) == 1:
                    occur_1 = True
                if swap_1_2 and int(e) == 2:
                    edge_matrix_str += '<edge_{}> '.format(1)
                elif swap_1_2 and int(e) == 1:
                    edge_matrix_str += '<edge_{}> '.format(2)
                else:
                    edge_matrix_str += '<edge_{}> '.format(int(e))
            elif int(e) == 3:
                edge_matrix_str += '<both_edges> '
                has_both_edges = True
            else:
                raise NotImplementedError
        # edge_matrix_str += ' '.join(str(int(e)) for e in graph.adj_matrix[i])
        # edge_matrix_str += ' '
        if i != len(graph.node_list) - 1:
            edge_matrix_str += ''
        else:
            edge_matrix_str += '<sep> '
    output += edge_matrix_str
    # if len(graph.node_list) == 6:
    # print(circuit_str)
    if has_both_edges:
        print('has both edges')
        print(inputs)
        print(output)
        input()
    return inputs, output

def parse_json_data_matrix_dutycycle_first(data_paths, output_path, target_vout=50, select_cond='none', use_log=False, six_comp=False):
    data_texts = []
    for data_path in data_paths:
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
        if six_comp:
            lower_threshold_eff = 0.0
        else:
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
            inputs, output = gen_textdata_from_raw_matrix_dutycycle_first(datum)
            # print(inputs)
            # print(output)
            # input()
            d_dict = {}
            # d_dict["instruction"] = instruction
            d_dict["d_cycle_option"] = [0.1, 0.3, 0.5, 0.7, 0.9]
            d_dict["vout"] = datum["vout"] / 100
            d_dict["eff"] = datum["eff"]
            d_dict["input"] = inputs
            d_dict["output"] = output
            data_text.append(d_dict)
        data_texts.append(data_text)
    data_text = list(itertools.chain(*data_texts))
    print("### Collect totally {} of data".format(len(data_text)))
    input('press enter to save')
    with open(output_path, 'w') as f:
        json.dump(data_text, f)


if __name__ == '__main__':
    args = get_transformer_args()
    config_path = 'parsers/config/parser.yaml'
    config = load_and_apply_yaml_config(config_path)
    os.makedirs(config.text_data_dir, exist_ok=True)

    data_path = 'dataset/raw/dataset_all_345_regenerate_prune_isomophic.json'
    output_dir = 'dataset/transformed'

    data_paths = [data_path]

    '''For 3-component circuit'''

    '''For SCFI'''
    # output_path = os.path.join(output_dir, 'dataset_345_shrink_canonical_typeNidx_dutycycle_first.json')
    # parse_json_data_shrink_canonical_dutycycle_first(data_paths=data_paths, output_path=output_path,
    #                 select_cond=args.select_cond, 
    #                 use_log=args.use_log,
    #                 target_vout=args.target_vout, typeNidx=True)
    
    '''For SCFI with output no component type'''
    # output_path = os.path.join(output_dir, 'dataset_345_shrink_canonical_typeNidx_output_no_type_dutycycle_first.json')
    # parse_json_data_shrink_canonical_dutycycle_first(data_paths=data_paths, output_path=output_path,
    #                 select_cond=args.select_cond, 
    #                 use_log=args.use_log,
    #                 target_vout=args.target_vout, typeNidx=True, output_no_type=True)
    # gen_volcabulary_canonical_typeNidx()

    '''For SFM'''
    output_path = os.path.join(output_dir, 'dataset_all_345_matrix_dutycycle_first.json')
    # output_path = os.path.join(output_dir, 'dataset_all_345_matrix_half_dutycycle_first.json')\\
    parse_json_data_matrix_dutycycle_first(data_paths=data_paths, output_path=output_path,
                    select_cond=args.select_cond, 
                    use_log=args.use_log,
                    target_vout=args.target_vout)
    # gen_volcabulary_matrix()

    '''For 6-component circuit'''

    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/new_dataset/6_component_raw/regenerate_30000'
    data_path = os.path.join(prefix, 'dataset_6_regenerate_30000_remove_isomophism.json')
    data_paths = [data_path]

    '''For SCFI'''
    # output_path = os.path.join(output_dir, 'dataset_6_regenerate_shrink_canonical_typeNidx.json')
    # parse_json_data_shrink_canonical_dutycycle_first(data_paths=data_paths, output_path=output_path,
    #                 select_cond=args.select_cond, 
    #                 use_log=args.use_log,
    #                 target_vout=args.target_vout,
    #                 six_comp=True, typeNidx=True)

    '''For SCFI with output no component type'''
    # output_path = os.path.join(output_dir, 'dataset_6_regenerate_shrink_canonical_typeNidx_output_no_type.json')
    # parse_json_data_shrink_canonical_dutycycle_first(data_paths=data_paths, output_path=output_path,
    #                 select_cond=args.select_cond, 
    #                 use_log=args.use_log,
    #                 target_vout=args.target_vout,
    #                 six_comp=True, typeNidx=True, output_no_type=True)
    
    '''For SFM'''
    # output_path = os.path.join(output_dir, 'dataset_6_regenerate_matrix_dutycycle_first.json')
    # output_path = os.path.join(output_dir, 'dataset_6_regenerate_matrix_half_dutycycle_first.json')
    # parse_json_data_matrix_dutycycle_first(data_paths=data_paths, output_path=output_path,
    #                 select_cond=args.select_cond, 
    #                 use_log=args.use_log,
    #                 target_vout=args.target_vout,
    #                 six_comp=True)
    

    exit()
    
    
    
    file_name = args.data.split('.')[0]

    feed_random_seeds(args.seed)

    if args.data is None:
        raise Exception('args.data not provided.')
    
    # output_dir = 
    
    parse_json_data(data_path=args.data, output_dir=output_path, select_cond=args.select_cond, use_log=args.use_log,
                                      target_vout=args.target_vout)

    # if args.use_log:
    #     wandb.init(project="surrogate_model",
    #                name=file_name,
    #                config=vars(args))
    if not args.data_seed:
        print('not using fixed dataset')
        args.data_seed = args.seed
    if not splitted_data_exist(data_path=args.data, training_ratio=args.train_ratio, seed=args.data_seed):
        # add 3,4 good circuits
        data = merge_extra_good_circuits(smaller_circuit_datasets=args.extra_datasets, _args=args,
                                         good_circuit_threshold=args.circuit_threshold)
        data = data + parse_json_data(data_path=args.data, select_cond=args.select_cond, use_log=args.use_log,
                                      target_vout=args.target_vout)

        data_train, data_dev, data_test = split_data(data=data,
                                                     data_path=args.data,
                                                     training_ratio=args.train_ratio,
                                                     dev_ratio=args.dev_ratio,
                                                     test_ratio=args.test_ratio,
                                                     split_by=args.split_by,
                                                     debug=args.debug,
                                                     seed=args.seed)
        data_train_file = f"{file_name}_train_{args.train_ratio}_{args.seed}.json"
        data_dev_file = f"{file_name}_dev_{args.train_ratio}_{args.seed}.json"
        data_test_file = f"{file_name}_test_{args.train_ratio}_{args.seed}.json"
        exit(0)

    else:
        print('keep using existing splitted data')
        data_train_file = f"{file_name}_train_{args.train_ratio}_{args.data_seed}.json"
        data_dev_file = f"{file_name}_dev_{args.train_ratio}_{args.data_seed}.json"
        data_test_file = f"{file_name}_test_{args.train_ratio}_{args.data_seed}.json"

    # ret_info = train_fn(args=args, training_data=data_train_file, validation_data=data_dev_file,
    #                     testing_data=data_test_file)
    # train_rse, valid_rse, test_rse = ret_info[-3:]

    # if args.use_log:
    #     wandb.run.summary["final_train_rse"] = train_rse
    #     wandb.run.summary["final_valid_rse"] = valid_rse
    #     wandb.run.summary["final_test_rse"] = test_rse

    # Plot simulation distribution
    # data_train = json.load(open(f"{file_name}_train_{args.train_ratio}_{args.seed}.json"))
    # distribution_plot(simulation=[dataum['reward'] for dataum in data_train],
    #                   predictions=[], file_name=f"train_{args.train_ratio}_{args.seed}.jpg", rse=0.0)
