import argparse
import sys
import os
import copy
import gc
# import networkx as nx
dir_path = os.getcwd()
sys.path.append(dir_path)

import itertools
import json
from matplotlib import pyplot as plt
import torch.nn as nn

import random
import numpy as np

# import wandb
import torch
from tqdm import tqdm

from transformer_args import get_transformer_args
from util import *
from threading import Thread
from multiprocessing import Process
import multiprocessing as mp
import networkx as nx

# sys.path.append(os.path.join(sys.path[0], '../topo_data_util/'))
# from train import main as train_fn
from topo_data_util.topo_analysis.topoGraph import TopoGraph
from GetReward import calculate_reward
from topo_data_util.topo_utils.plot import plot_hist
from utils.yaml_parser import load_and_apply_yaml_config
from parsers.simulation import sim_generation_output, read_LLM_ouput, read_masked_LLM_output, sim_masked_generation_output, convert_netlist_2_graph
from parsers.data_utils import *
from analog_LLM.utils.utils import random_split_trn_val

def sim(raw_data_split, modified_data, path, i):
    for name, datum in tqdm(raw_data_split[i].items()):
        instruction, input_ids, output = gen_textdata_from_raw(datum)
        if os.path.exists(path):
            os.remove(path)
        # path = 'sim_data.cki'
        # output = "Here's the circuit representation using a hypergraph: Vertices:C1, VIN, GND, Sa0, C0, VOUT, Sb0, Sb1 Hyperedges:(VOUT, Sb1), (VIN, Sa0, C0, C1), (Sa0, Sb1, Sb0), (GND, C1), (C0, Sb0) The duty cycle is set to 0.3."
        print(path)
        result = sim_generation_output(path, input_ids, output)
        print(result)
        datum["vout"] = result['Vout']
        datum["eff"] = result['efficiency']
        # print(datum)
        modified_data[i][name] = datum
        
def split_data_four_way(data_path, output_path):
    print('loading from json file')
    raw_data = json.load(open(data_path, 'r'))
    total_d_num_1 = 0
    for datum in tqdm(raw_data):
        total_d_num_1 += 1
    print('total_d_num_1', total_d_num_1)
    # input()
    data1 = []
    data2 = []
    data3 = []
    data4 = []
    n = 0
    for datum in tqdm(raw_data):
        n += 1
        if n < total_d_num_1 / 4:
            data1.append(datum)
        elif n >= total_d_num_1 / 4 and n < total_d_num_1 / 2:
            data2.append(datum)
        elif n >= total_d_num_1 / 2 and n < total_d_num_1 /4*3:
            data3.append(datum)
        else:
            data4.append(datum)
    print(len(data1), len(data2), len(data3), len(data4))
    with open(os.path.join(output_path, 'data1.json'), 'w') as f:
        json.dump(data1, f)
    with open(os.path.join(output_path, 'data2.json'), 'w') as f:
        json.dump(data2, f)
    with open(os.path.join(output_path, 'data3.json'), 'w') as f:
        json.dump(data3, f)
    with open(os.path.join(output_path, 'data4.json'), 'w') as f:
        json.dump(data4, f)
    

def regenerate_data_with_sim(prefix, data_path, output_path, portion_idx):

    print('loading from json file:', data_path)
    raw_data = json.load(open(data_path, 'r'))
    total_d_num_1 = 0
    for datum in tqdm(raw_data):
        total_d_num_1 += 1
    n_thread = 11
    total_d_num_1 = total_d_num_1 * 4 # because number of duty cycle options is 4
    d_num_per_threads = int(total_d_num_1 / (n_thread)-1)
    # d_num_per_threads = int(5/(n_thread))
    # total_d_num_1 = 5
    print("total_d_num_1", total_d_num_1)
    print('d_num_per_threads', d_num_per_threads)
    data_splits = []
    n = 0
    total_d_num = 0
    data_split = {}
    
    for datum in tqdm(raw_data):
        duty_cycle_options = [0.2, 0.4, 0.6, 0.8]
        for duty_cycle in duty_cycle_options:
            datum_new = {}
            datum_new["duty_cycle"] = duty_cycle
            datum_new["list_of_node"] = datum["list_of_node"]
            datum_new["list_of_edge"] = datum["list_of_edge"]
            datum_new["vout"] = 1.0
            datum_new["eff"] = 1.0
            name_new = str(n) + '_' + str(duty_cycle)
            n += 1
            data_split[name_new] = datum_new
            # print(n)
            if n % d_num_per_threads == 0:
                data_splits.append(data_split)
                total_d_num += len(data_split)
                # print("add", len(data_split), "to data_splits")
                data_split = {}
                if n == total_d_num_1:
                    print('end of split ... ')
                    break
            elif n == total_d_num_1:
                data_splits.append(data_split)
                total_d_num += len(data_split)
                print('end of split ... ')
                break
    print('total_d_num', total_d_num)
    assert(total_d_num == total_d_num_1)
    print(len(data_splits))
    assert(len(data_splits) == n_thread)
    manager = mp.Manager()
    modified_data_splits = []
    for i in range(n_thread):
        modified_data_splits.append(manager.dict())

    # modified_data_splits = copy.deepcopy(data_splits)
    # print('before simulation', data_splits)
    # input()
    del raw_data
    gc.collect()
    threads = list()
    os.makedirs(os.path.join(prefix, 'data{}'.format(portion_idx)), exist_ok=True)
    for i in range(n_thread):
        path = os.path.join(prefix, 'data{}/sim{}.cki'.format(portion_idx, str(i))) 
        thread = Process(target=sim, args=(data_splits, modified_data_splits, path, i))
        # thread = Process(target=sim, args=(data_splits[i], modified_data_splits[i], path))
        threads.append(thread)
    print(f'Created {len(threads)} processes')
    
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
        
    # print('\n\ndata_splits')
    # print(data_splits)
    del data_splits
    gc.collect()
    
    # print('\nmodified_data_splits')
    # print(modified_data_splits[i])
    modified_data = []
    for data_dict in tqdm(modified_data_splits):
        for name, datum in data_dict.items():
            modified_data.append(datum)
        
    # print('\nmodified_data', modified_data)
    with open(output_path, 'w') as f:
        json.dump(modified_data, f)
    print(len(modified_data))

def combine_split_data(prefix):
    def merge_dicts(*dict_args):
        """
        Given any number of dictionaries, shallow copy and merge into a new dict,
        precedence goes to key-value pairs in latter dictionaries.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result
    dicts = []
    result = {}
    total_data = []
    # prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/remove_redundant_circuit'
    num = 0
    for i in range(1, 5):
        data_name = "dataset_dutycycle_regenerate_{}.json".format(str(i))
        data_path = os.path.join(prefix, data_name)
        raw_data = json.load(open(data_path, 'r'))
        # raw_data = list(raw_data.values())
        raw_data_dict = {}
        for datum in raw_data:
            raw_data_dict[str(num)] = datum
            num += 1
        # print(type(raw_data))
        # total_data = total_data + raw_data
        # input()
        result = merge_dicts(result, raw_data_dict)
        print('current len of result', len(result))
    output_path = os.path.join(prefix, 'dataset_dutycycle_regenerate_all.json')
    result = list(result.values())
    print(result[2])
    with open(output_path, 'w') as f:
        json.dump(result, f)
    valid_num = 0
    for datum in result:
        if datum["eff"] == -1:
            continue
        valid_num += 1
    print('valid_num', valid_num)

def get_clip_threshold(data_power_conversion, data_efficiency):
    upper_threshold_power = np.percentile(data_power_conversion, 99.5)
    lower_threshold_power = np.percentile(data_power_conversion, 0.5)
    upper_threshold_eff = np.percentile(data_efficiency, 99.5)
    lower_threshold_eff = np.percentile(data_efficiency, 0.5)
    return upper_threshold_power, lower_threshold_power, upper_threshold_eff, lower_threshold_eff

def extract_vout_eff(raw_data):
    data_power_conversion = []
    data_efficiency = []
    voltage2data = {}
    idx = 0
    for name, datum in tqdm(raw_data.items()):
        data_power_conversion.append(datum["vout"])
        data_efficiency.append(datum["eff"])
        # print(datum)
        # input()
        list_of_node = []
        for node in datum['list_of_node']:
            if type(node) == int:
                continue
            list_of_node.append(node)
            # print(list_of_node)
        # print(list_of_node)
        list_of_node.sort()
        node_str = ''
        for node in list_of_node:
            node_str += (' '+node)
        # if (list_of_node, datum["vout"], datum["eff"]) == (['C0', 'GND', 'L0', 'L1', 'Sa0', 'Sb0', 'VIN', 'VOUT'], 19, 0.19):
        # if (list_of_node, datum["vout"], datum["eff"]) == (['C0', 'GND', 'L0', 'Sa0', 'Sa1', 'Sb0', 'VIN', 'VOUT'], 1, 0.01):
        # if (node_str, datum["vout"], datum["eff"]) in voltage2data.keys():
        #     print(name)
        #     print(datum)
        #     path = 'sim_check.cki'
        #     instruction, input_ids, output = gen_textdata_from_raw(datum)
        #     print(output)
        #     result = sim_generation_output(path, output)
        #     datum["vout"] = result['Vout']
        #     datum["eff"] = result['efficiency']
        #     print(result)
        #     # print(datum)
        #     input()
        if (node_str, datum["vout"], datum["eff"]) in voltage2data.keys():
            # print(voltage2data.keys())
            # print(idx)
            # print((list_of_node, datum["vout"], datum["eff"]))
            voltage2data[(node_str, datum["vout"], datum["eff"])].append(datum)
            # input()
        else:
            voltage2data[(node_str, datum["vout"], datum["eff"])] = [datum]
        idx += 1
    print('voltage2data', len(voltage2data))
    print('raw_data', len(raw_data))
    input()
    for name, datums in tqdm(voltage2data.items()):
        if len(datums) == 1:
            continue
        idx = 0
        graphs = []
        for datum in datums:
            if datum['eff'] != -1:
                continue
            # if datum["list_of_edge"] == [['C0', 12], ['VIN', 5], ['Sa0', 11], ['Sb0', 6], ['GND', 9], ['L1', 12], ['L0', 12], ['L0', 11], ['Sb0', 5], ['L1', 9], ['C0', 9], ['VOUT', 6], ['Sa0', 6]]:
            path = 'sim_check.cki'
            # datum["list_of_edge"] = [['VIN', 11], ['L0', 9], [9, 'GND'], ['VOUT', 7],  ['Sb0', 11], ['L0', 11], ['Sb0', 7]]
            # datum["list_of_node"] = ['GND', 7, 'Sb0', 9, 11, 'VIN', 'L0', 'VOUT']
            # datum["list_of_edge"] = [['VIN', 11], ['Sb0', 11], ['Sb0', 7],  ['VOUT', 7]]
            # datum["list_of_node"] = ['Sb0', 11, 'VIN', 7, 'VOUT']
            # 'vout': 95.77336907345614, 'eff': 0.8976705316585513,
            instruction, input_ids, output = gen_textdata_from_raw(datum)
            print(datum)
            # print(output)
            result = sim_generation_output(path, output)
            datum["vout"] = result['Vout']
            datum["eff"] = result['efficiency']
            print(result)
            input()
            topo_file = os.path.join('plot', 'figure{}.png'.format(idx))
            T = nx.Graph()
            for node in datum["list_of_node"]:
                if type(node) == int:
                    T.add_node(node, type='connection')
                elif node == 'VIN' or node == 'VOUT' or node == 'GND':
                    T.add_node(node, type=node)
                else:
                    T.add_node(node, type=node[:len(node)-1])
            # T.add_nodes_from((datum["list_of_node"]))
            T.add_edges_from(datum["list_of_edge"])
            graphs.append(T)
            plt.figure()
            nx.draw(T, with_labels=True)
            plt.savefig(topo_file)
            # T.clear()
            plt.close()
            print(datum)
            idx += 1
        # print(graphs[0])
        # print(graphs[1])
        # print(nx.vf2pp_is_isomorphic(graphs[0], graphs[1], node_label='type'))
        # input()
    print("finish")
    input()
    data_power_conversion = np.array(data_power_conversion)
    print(data_power_conversion)
    # data_power_conversion = np.around(data_power_conversion, decimals=2)
    data_efficiency = np.array(data_efficiency)
    print(data_efficiency)
    # data_efficiency = np.around(data_efficiency, decimals=2)
    
    upper_threshold_power, lower_threshold_power, upper_threshold_eff, lower_threshold_eff = get_clip_threshold(data_power_conversion, data_efficiency)

    vout_data = np.clip(data_power_conversion, lower_threshold_power, upper_threshold_power)/100
    eff_data = np.clip(data_efficiency, lower_threshold_eff, upper_threshold_eff)
    return vout_data, eff_data

def check_sim_regenerate():
    # prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
    # # data_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate.json')
    # data_path = os.path.join(prefix, 'data4_regenerate_try.json')
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4'
    data_path = os.path.join(prefix, 'data1_regenerate.json')
    raw_data = json.load(open(data_path, 'r'))
    vout_data, eff_data = extract_vout_eff(raw_data)

    # print(len(vout_data[vout_data<0.5]))
    # print(len(vout_data[vout_data>=0.5]))

    threshold = np.linspace(0.0, 1.0, num=101)
    # print(threshold)
    num = 0
    trn_num = 0
    val_num = 0
    for idx, thred in enumerate(threshold):
        if idx == 0:
            index = np.where(vout_data<=thred)[0]
            print(vout_data[index])
        else:
            index = np.where(np.logical_and(vout_data>threshold[idx-1], vout_data<=thred))[0]
            # print(index)
        if (idx+1)%5 == 0:
            print(thred)
            val_num += len(index)
        else:
            trn_num += len(index)
        if idx == len(threshold) - 1:
            index = np.where(vout_data>thred)[0]
            trn_num += len(index)
        num = num + 1
    print('trn_num', trn_num)
    print('val_num', val_num)

    plt.hist(vout_data, bins=200)
    plt.savefig('plot/vout_hist_regenerate.png',  dpi=200)
    plt.close()
    plt.hist(eff_data, bins=100)
    plt.savefig('plot/eff_hist_regenerate.png',  dpi=200)
    print('save hist ...')
    input()

    print(len(raw_data))
    data = []
    for name, datum in tqdm(raw_data.items()):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            continue
        vout, eff = sim_single_data(datum, prefix)
        print('original vout and eff: ', datum["vout"], ', ', datum["eff"])
        print('simulate vout and eff: ', vout, ', ', eff)
        input()
        data.append(datum)

def remove_isomorphic_circuit_main():
    data_dir = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/new_dataset/6_component_raw/regenerate_30000/"
    data_path = data_dir + "dataset_6_regenerate_30000_all.json"
    output_path = data_dir + "dataset_6_regenerate_30000_remove_isomophism.json"
    print("[RUNNING] remove_isomorphic_circuit_main for {}-component cirucits with data_path={} and output_path={}".format(7, data_path, output_path))
    raw_data = json.load(open(data_path, 'r'))
    new_raw_data = remove_isomorphism_circuit(raw_data)
    print('finish remove_isomorphism_circuit, press ENTER to save data')
    input()
    with open(output_path, 'w') as f:
        json.dump(new_raw_data, f)

def convert_raw_2_matrix_main():
    print("RUNNING convert_raw_2_matrix_main")
    # prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
    # data_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_prune_isomophic.json')
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/new_dataset/6_component_raw/regenerate_30000'
    data_path = os.path.join(prefix, 'dataset_6_regenerate_30000_remove_isomophism.json')
    raw_data = json.load(open(data_path, 'r'))
    matrix_data = convert_raw_2_matrix(raw_data)
    print('finish remove_isomorphism_circuit, press ENTER to save data')
    input()
    output_prefix = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/text_dataset/masked"
    data_path = os.path.join(output_prefix, 'dataset_6_regenerate.json')
    # data_path = os.path.join(output_prefix, 'dataset_6_regenerate_graph_first.json')
    with open(data_path, 'w') as f:
        json.dump(matrix_data, f)

def split_isomorphic_netlist(data_path, output_path):
    raw_data = json.load(open(data_path, 'r'))

    trn_graphs = []
    data_ids = []
    for idx, datum in enumerate(tqdm(raw_data)):
        graph = nx.Graph()
        for node in datum["list_of_node"]:
            if type(node) == int:
                graph.add_node(node, type='connection')
            elif node == 'VIN' or node == 'VOUT' or node == 'GND':
                graph.add_node(node, type=node)
            else:
                graph.add_node(node, type=node[:len(node)-1])
        graph.add_edges_from(datum["list_of_edge"])
        if len(trn_graphs) == 0:
            trn_graphs.append(graph)
            new_data = {}
            new_data["list_of_edge"] = datum["list_of_edge"]
            new_data["list_of_node"] = datum["list_of_node"]
            data_ids.append(new_data)
        else:
            is_isomorphic = False
            for i, trn_graph in enumerate(trn_graphs):
                if nx.vf2pp_is_isomorphic(trn_graph, graph, node_label='type'):
                    is_isomorphic = True
                    break
            if not is_isomorphic:
                trn_graphs.append(graph)
                new_data = {}
                new_data["list_of_edge"] = datum["list_of_edge"]
                new_data["list_of_node"] = datum["list_of_node"]
                data_ids.append(new_data)
    with open(output_path, 'w') as f:
        json.dump(data_ids, f)

def convert_raw_2_matrix_main():
    print("RUNNING convert_raw_2_matrix_main")
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_all_component345'
    data_path = os.path.join(prefix, 'dataset_all_345_regenerate_prune_isomophic.json')
    raw_data = json.load(open(data_path, 'r'))
    matrix_data = convert_raw_2_matrix(raw_data)
    print('finish remove_isomorphism_circuit, press ENTER to save data')
    input()
    output_prefix = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/text_dataset/masked"
    # data_path = os.path.join(output_prefix, 'dataset_all_345_regenerate_prune_isomophic.json')
    # data_path = os.path.join(output_prefix, 'dataset_all_345_regenerate_prune_isomophic_new.json')
    data_path = os.path.join(output_prefix, 'dataset_all_345_regenerate_prune_isomophic_new_graphfirst.json')
    with open(data_path, 'w') as f:
        json.dump(matrix_data, f)

def generate_345_component_circuit_data():
    '''For 345-component circuits'''
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_all_component345'
    original_data_path = os.path.join(prefix, 'dataset_all_345_regenerate_prune_isomophic.json')
    netlist_data_path = os.path.join(prefix, 'netlist_345_nonisomorphic.json')
    split_isomorphic_netlist(original_data_path, netlist_data_path)

    prefix = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_all_component345/duty_cycle_regenerate"
    split_data_four_way(netlist_data_path, prefix)
    # running in 2447 tmux 3 running
    portion_idx = 1
    # running in 2413 tmux 1 running
    portion_idx = 2
    # # # running in 2412 tmux 1 running
    portion_idx = 3
    # # # running in 2420 tmux 1 running
    portion_idx = 4
    data_path = os.path.join(prefix, f'data{portion_idx}.json')
    output_path = os.path.join(prefix, f'dataset_dutycycle_regenerate_{portion_idx}.json')
    regenerate_data_with_sim(prefix=prefix, data_path=data_path, output_path=output_path, portion_idx=portion_idx)
    combine_split_data(prefix)

def generate_6_component_circuit_data():
    '''For 6-component circuits'''
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/new_dataset/6_component_raw/regenerate_30000'
    original_data_path = os.path.join(prefix, 'dataset_6_regenerate_30000_remove_isomophism.json')

    netlist_data_path = os.path.join(prefix, 'netlist_6_nonisomorphic.json')
    # split_isomorphic_netlist(original_data_path, netlist_data_path)
    
    prefix = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/new_dataset/6_component_raw/regenerate_30000/duty_cycle_regenerate"
    os.makedirs(prefix, exist_ok=True)
    split_data_four_way(netlist_data_path, prefix)
    # running in 2447 tmux 6 running
    portion_idx = 1
    # running in 2413 tmux 1 running
    # portion_idx = 2
    # # # running in 2412 tmux 1 running
    # portion_idx = 3
    # # # running in 2420 tmux 1 running
    # portion_idx = 4
    data_path = os.path.join(prefix, f'data{portion_idx}.json')
    output_path = os.path.join(prefix, f'dataset_dutycycle_regenerate_{portion_idx}.json')
    regenerate_data_with_sim(prefix=prefix, data_path=data_path, output_path=output_path, portion_idx=portion_idx)
    # combine_split_data(prefix)

if __name__ == '__main__':
    generate_345_component_circuit_data()
    # run in 2447 tmux 6
    # generate_6_component_circuit_data()
    
    exit()

    '''
    [FINISHED] data4_regenerate_try run in ssh -p 2424 skunk@50.22.159.227
    data1_regenerate_try run in ssh -p 2417 skunk@50.22.159.227

    [FINISHED]
    ssh -p 2410 skunk@50.22.159.227
    running regenerate_data_with_sim with portion_idx=1 component3 circuit
    running regenerate_data_with_sim with portion_idx=2 component3 circuit
    ssh -p 2424 skunk@50.22.159.227
    running regenerate_data_with_sim with portion_idx=3 component3 circuit
    ssh -p 2449 skunk@50.22.159.227
    running regenerate_data_with_sim with portion_idx=4 component3 circuit

    ssh -p 2449 skunk@50.22.159.227
    running regenerate_data_with_sim with portion_idx=1 component4 circuit
    ssh -p 2417 skunk@50.22.159.227
    running regenerate_data_with_sim with portion_idx=2 component4 circuit
    ssh -p 2424 skunk@50.22.159.227
    running regenerate_data_with_sim with portion_idx=3 component4 circuit
    '''
    
    