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
from parsers.simulation import sim_generation_output, read_LLM_ouput, read_masked_LLM_output, sim_masked_generation_output, convert_netlist_2_graph, sim_netlist_duty_cycle_L
from parsers.data_utils import *
from analog_LLM.utils.utils import random_split_trn_val

def sim(raw_data_split, modified_data, path, i):
    for name, datum in tqdm(raw_data_split[i].items()):
        instruction, input_ids, output = gen_textdata_from_raw(datum)
        # path = 'sim_data.cki'
        # output = "Here's the circuit representation using a hypergraph: Vertices:C1, VIN, GND, Sa0, C0, VOUT, Sb0, Sb1 Hyperedges:(VOUT, Sb1), (VIN, Sa0, C0, C1), (Sa0, Sb1, Sb0), (GND, C1), (C0, Sb0) The duty cycle is set to 0.3."
        result = sim_generation_output(path, output)
        print(result)
        datum["vout"] = result['Vout']
        datum["eff"] = result['efficiency']
        # print(datum)
        modified_data[i][name] = datum
        
def split_data_four_way(data_path):
    output_dir = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4"
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523'
    data_path = os.path.join(prefix, 'dataset_4.json')
    print('loading from json file')
    raw_data = json.load(open(data_path, 'r'))
    total_d_num_1 = 0
    for name, datum in tqdm(raw_data.items()):
        # if "valid" in datum.keys():
        #     is_valid = datum["valid"]
        # elif "label" in datum.keys():
        #     is_valid = datum["label"]
        # else:
        #     is_valid = 0 if datum["eff"] == 0 else 1  # if no valid label, assume it's valid
        # if is_valid == 0:
        #     continue
        total_d_num_1 += 1
    print('total_d_num_1', total_d_num_1)
    # input()
    data1 = {}
    data2 = {}
    data3 = {}
    data4 = {}
    n = 0
    for name, datum in tqdm(raw_data.items()):
        # if "valid" in datum.keys():
        #     is_valid = datum["valid"]
        # elif "label" in datum.keys():
        #     is_valid = datum["label"]
        # else:
        #     is_valid = 0 if datum["eff"] == 0 else 1  # if no valid label, assume it's valid
        # if is_valid == 0:
        #     continue
        n += 1
        if n < total_d_num_1 / 4:
            data1[name] = datum
        elif n >= total_d_num_1 / 4 and n < total_d_num_1 / 2:
            data2[name] = datum
        elif n >= total_d_num_1 / 2 and n < total_d_num_1 /4*3:
            data3[name] = datum
        else:
            data4[name] = datum
    print(len(data1), len(data2), len(data3), len(data4))
    with open(os.path.join(output_dir, 'data1.json'), 'w') as f:
        json.dump(data1, f)
    with open(os.path.join(output_dir, 'data2.json'), 'w') as f:
        json.dump(data2, f)
    with open(os.path.join(output_dir, 'data3.json'), 'w') as f:
        json.dump(data3, f)
    with open(os.path.join(output_dir, 'data4.json'), 'w') as f:
        json.dump(data4, f)
    

def regenerate_data_with_sim(prefix, data_path, output_path, portion_idx):

    print('loading from json file')
    raw_data = json.load(open(data_path, 'r'))
    total_d_num_1 = 0
    for name, datum in tqdm(raw_data.items()):
        # if "valid" in datum.keys():
        #     is_valid = datum["valid"]
        # elif "label" in datum.keys():
        #     is_valid = datum["label"]
        # else:
        #     is_valid = 0 if datum["eff"] == 0 else 1  # if no valid label, assume it's valid
        # if is_valid == 0:
        #     continue
        total_d_num_1 += 1
    n_thread = 32
    d_num_per_threads = int(total_d_num_1 / (n_thread-1))
    # d_num_per_threads = int(5/(n_thread))
    # total_d_num_1 = 5
    print("total_d_num_1", total_d_num_1)
    print('d_num_per_threads', d_num_per_threads)
    data_splits = []
    n = 0
    total_d_num = 0
    data_split = {}
    
    for name, datum in tqdm(raw_data.items()):
        # if "valid" in datum.keys():
        #     is_valid = datum["valid"]
        # elif "label" in datum.keys():
        #     is_valid = datum["label"]
        # else:
        #     is_valid = 0 if datum["eff"] == 0 else 1  # if no valid label, assume it's valid
        # if is_valid == 0:
        #     continue
        n += 1
        # datum["vout"] = 1.0
        # datum["eff"] = 1.0
        data_split[name] = datum
        # print(n)
        if n % d_num_per_threads == 0:
            data_splits.append(data_split)
            total_d_num += len(data_split)
            data_split = {}
            if n == total_d_num_1:
                print('end of split ... ')
                break
            # print("add")
        elif n == total_d_num_1:
            data_splits.append(data_split)
            total_d_num += len(data_split)
            print('end of split ... ')
            break
    assert(total_d_num == total_d_num_1)
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
    
    for i in range(n_thread):
        path = os.path.join(prefix, 'misc{}/sim{}.cki'.format(portion_idx, str(i))) 
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
    modified_data = {}
    for data_dict in tqdm(modified_data_splits):
        for name, datum in data_dict.items():
            modified_data[name] = datum
        
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
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/remove_redundant_circuit'
    num = 0
    for i in range(3, 6):
        data_name = "data{}_regenerate.json".format(str(i))
        data_name = "dataset_{}_remove_redundant.json".format(str(i))
        data_path = os.path.join(prefix, data_name)
        raw_data = json.load(open(data_path, 'r'))
        
        raw_data_dict = {}
        for datum in raw_data:
            raw_data_dict[str(num)] = datum
            num += 1
        # print(type(raw_data))
        # total_data = total_data + raw_data
        # input()
        result = merge_dicts(result, raw_data_dict)
        print('current len of result', len(result))
    # output_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_2.json')
    output_path = os.path.join(prefix, 'dataset_345_remove_redundant.json')
    with open(output_path, 'w') as f:
        json.dump(result, f)

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

def parse_json_data(data_path, output_path, target_vout=50, select_cond='none', use_log=False):
    """
    Convert the dataset json to formats that can be loaded by transformer

    :param data_path: path of the raw data
    :param target_vout: target output voltage for reward
    :param select_cond: 'max_reward' finds the one with the highest reward;
       'fix_cycle' finds the one with a fixed duty cycle (0.5).
    """
    print('loading from json file')
    raw_data = json.load(open(data_path, 'r'))
    data = []
    
    data_power_conversion = []
    data_efficiency = []
    for name, datum in tqdm(raw_data.items()):
        data_power_conversion.append(datum["vout_analytic"])
        data_efficiency.append(datum["eff"])
    data_power_conversion = np.array(data_power_conversion)
    upper_threshold_power = np.percentile(data_power_conversion, 99.5)
    lower_threshold_power = np.percentile(data_power_conversion, 0.5)
    
    data_efficiency = np.array(data_efficiency)
    upper_threshold_eff = np.percentile(data_efficiency, 99.5)
    lower_threshold_eff = np.percentile(data_efficiency, 0.5)
        
    data_text = []
    # data_power_conversion = []
    # data_efficiency = []
    print('processing data')
    for name, datum in tqdm(raw_data.items()):
        d_dict={}
        if datum["vout_analytic"] >= upper_threshold_power or datum["vout_analytic"] <= lower_threshold_power:
            continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            continue
        instruction, input_ids, output = gen_textdata_from_raw(datum)
        # instruction, input_ids, output = gen_textdata_topo2power_from_raw(datum)
        # # print(instruction)
        # # print(input_ids)
        # # print(output)
        # # input()
        d_dict["instruction"] = instruction
        d_dict["input"] = input_ids
        d_dict["output"] = output
        data_text.append(d_dict)
        data.append(datum)
    
    print("### Collect totally {} of data".format(len(data_text)))
    jdump(data_text, os.path.join(output_path))
    
def remove_isomorphic_circuit_main():
    n_component = 3
    if n_component == 5:
        # Parameters for 5-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
        data_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_2.json')
        output_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_prune_isomophic.json')
    elif n_component == 4:
        # Parameters for 4-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4'
        data_path = os.path.join(prefix, "dataset_4_regenerate.json")
        output_path = os.path.join(prefix, 'dataset_4_valid_set_regenerate_prune_isomophic.json')
    elif n_component == 3:
        # Parameters for 3-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component3'
        data_path = os.path.join(prefix, "dataset_3_regenerate.json")
        output_path = os.path.join(prefix, 'dataset_3_valid_set_regenerate_prune_isomophic.json')
    else:
        raise NotImplementedError

    print("[RUNNING] remove_isomorphic_circuit_main for {}-component cirucits with data_path={} and output_path={}".format(n_component, data_path, output_path))
    raw_data = json.load(open(data_path, 'r'))
    new_raw_data = remove_isomorphism_circuit(raw_data)
    print('finish remove_isomorphism_circuit, press ENTER to save data')
    input()
    with open(output_path, 'w') as f:
        json.dump(new_raw_data, f)

def remove_isomorphic_circuit_main_all_345_component_data():
    print("[RUNNING] remove_isomorphic_circuit_main_all_345_component_data")
    # Parameters for 5-component circuit
    print('### loading 5-component data')
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
    data_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_prune_isomophic.json')
    raw_data_all = json.load(open(data_path, 'r'))
    assert(type(raw_data_all) == list)
    # Parameters for 4-component circuit
    print('### loading 4-component data')
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4'
    data_path = os.path.join(prefix, 'dataset_4_valid_set_regenerate_prune_isomophic.json')
    raw_data_all = raw_data_all + json.load(open(data_path, 'r'))
    # Parameters for 3-component circuit
    print('### loading 3-component data')
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component3'
    data_path = os.path.join(prefix, "dataset_3_valid_set_regenerate_prune_isomophic.json")
    raw_data_all = raw_data_all + json.load(open(data_path, 'r'))
    # Parameters for 345-component remove-redundant circuit
    print("### loading 345-component remove-redundant circuit")
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/remove_redundant_circuit'
    data_path = os.path.join(prefix, 'dataset_345_remove_redundant_regenerate.json')
    raw_data = json.load(open(data_path, 'r'))
    raw_data_all = raw_data_all + list(raw_data.values())
    print("### Collect totally {} of data".format(len(raw_data_all)))
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_all_component345'
    os.makedirs(prefix, exist_ok=True)
    output_path = os.path.join(prefix, 'dataset_all_345_regenerate_prune_isomophic.json')
    new_raw_data = remove_isomorphism_circuit(raw_data_all)
    print('finish remove_isomorphism_circuit, press ENTER to save data')
    input()
    with open(output_path, 'w') as f:
        json.dump(new_raw_data, f)

    
    
def check_remove_isomorphic_circuit_main():
    print("RUNNING check_remove_isomorphic_circuit_main")
    n_component = 5
    if n_component == 5:
        # Parameters for 5-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
        data_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_prune_isomophic.json')
    elif n_component == 4:
        # Parameters for 4-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4'
        data_path = os.path.join(prefix, 'dataset_4_valid_set_regenerate_prune_isomophic.json')
    elif n_component == 3:
        # Parameters for 3-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component3'
        data_path = os.path.join(prefix, "dataset_3_valid_set_regenerate_prune_isomophic.json")
    else:
        raise NotImplementedError
    raw_data = json.load(open(data_path, 'r'))
    remove_isomorphism_circuit(raw_data, check_isomorphism=True)


def check():
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_all_component345'
    data_path = os.path.join(prefix, 'dataset_all_345_regenerate_prune_isomophic.json')
    raw_data = json.load(open(data_path, 'r'))
    upper_threshold_power = 55
    lower_threshold_power = 45
    upper_threshold_eff = 1
    lower_threshold_eff = 0.9
    for datum in tqdm(raw_data):
        if datum["vout"] >= upper_threshold_power or datum["vout"] <= lower_threshold_power:
            continue
        if datum["eff"] >= upper_threshold_eff or datum["eff"] <= lower_threshold_eff:
            continue
        
        print('datum: ', datum)
        input()
        vout, eff = sim_single_data(datum, prefix)
        print('original vout and eff: ', datum["vout"], ', ', datum["eff"])
        print('simulate vout and eff: ', vout, ', ', eff)
        input()

def process_data_3():
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523'
    data_path = os.path.join(prefix, 'dataset_3.json')
    raw_data = json.load(open(data_path, 'r'))
    print(len(raw_data))
    num = 0
    for name, datum in tqdm(raw_data.items()):
        if "valid" in datum.keys():
            is_valid = datum["valid"]
        elif "label" in datum.keys():
            is_valid = datum["label"]
        else:
            is_valid = 0 if datum["eff"] == 0 else 1  # if no valid label, assume it's valid
        if is_valid == 0:
            print(datum)
            instruction, input_ids, output = gen_textdata_from_raw(datum)
            result = sim_generation_output('sim_check.cki', output)
            print(result)
            topo_file = os.path.join('plot', 'figure3_{}.png'.format(num))
            T = nx.Graph()
            for node in datum["list_of_node"]:
                if type(node) == int:
                    T.add_node(node, type='connection')
                elif node == 'VIN' or node == 'VOUT' or node == 'GND':
                    T.add_node(node, type=node)
                else:
                    T.add_node(node, type=node[:len(node)-1])
            T.add_edges_from(datum["list_of_edge"])
            plt.figure()
            nx.draw(T, with_labels=True)
            plt.savefig(topo_file)
            # T.clear()
            plt.close()
            input()
            continue
        num+=1
    print(num)

def extract_essensial_remove_redundant_circuit_main():
    n_component = 3
    out_prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/remove_redundant_circuit'
    if n_component == 5:
        # Parameters for 5-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
        data_path = os.path.join(prefix, 'dataset_5_valid_set_regenerate_prune_isomophic.json')
        output_path = os.path.join(out_prefix, 'dataset_5_remove_redundant.json')
    elif n_component == 4:
        # Parameters for 4-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4'
        data_path = os.path.join(prefix, "dataset_4_valid_set_regenerate_prune_isomophic.json")
        output_path = os.path.join(out_prefix, 'dataset_4_remove_redundant.json')
    elif n_component == 3:
        # Parameters for 3-component circuit
        prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component3'
        data_path = os.path.join(prefix, "dataset_3_valid_set_regenerate_prune_isomophic.json")
        output_path = os.path.join(out_prefix, 'dataset_3_remove_redundant.json')
    else:
        raise NotImplementedError
    print("[RUNNING] extract_essensial_remove_redundant_circuit_main for {}-component cirucits with data_path={} and output_path={}".format(n_component, data_path, output_path))
    raw_data = json.load(open(data_path, 'r'))
    datums_new = extract_essensial_remove_redundant_circuit(raw_data)
    with open(output_path, 'w') as f:
        json.dump(datums_new, f)

def test_masked_simulation():
    print("RUNNING test_masked_simulation")
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_all_component345'
    data_path = os.path.join(prefix, 'dataset_all_345_regenerate_prune_isomophic.json')
    raw_data = json.load(open(data_path, 'r'))
    # for datum in tqdm(raw_data):
    node_tokens = set()
    type_str = ['Sa', 'Sb', 'C', 'L']
    for device in type_str:
        for i in range(5):
            device_str = device + str(i)
            node_tokens.add(device_str)
    node_tokens.add('IN')
    node_tokens.add('OUT')
    node_tokens.add('0')
    for i in range(0, len(raw_data), 5):
        datum = raw_data[i]
        # print(datum)
        _, _, output = gen_textdata_from_raw(datum)
        netlist1, duty_cycle1 = read_LLM_ouput(output)
        graph1 = convert_netlist_2_graph(node_tokens, netlist1)
        d_dict = gen_adjmatrix_textdata_from_raw_for_test(datum)
        # print('vout', d_dict['vout'], 'eff', d_dict['eff'])
        # result = sim_masked_generation_output('sim_check.cki', d_dict['circuit_str'])
        # print(result)
        # print('circuit_str', d_dict['circuit_str'])
        # input()
        netlist2, duty_cycle2 = read_masked_LLM_output(d_dict['circuit_str'])
        graph2 = convert_netlist_2_graph(node_tokens, netlist2)
        # if not nx.vf2pp_is_isomorphic(graph1, graph2, node_label='type'):
        # if not nx.vf2pp_is_isomorphic(graph1, graph2, node_label='type'):
        if i == 215:
            print(i)
            print(datum)
            print('vout', d_dict['vout'], 'eff', d_dict['eff'])
            print('circuit_str', d_dict['circuit_str'])
            print('netlist1', netlist1)
            print('netlist2', netlist2)
            plt.figure()
            nx.draw(graph1, with_labels=True)
            plt.savefig('sample.png')
            assert(nx.vf2pp_is_isomorphic(graph1, graph2, node_label='type') == True)
            input()
        assert(nx.vf2pp_is_isomorphic(graph1, graph2, node_label='type') == True)
        assert(duty_cycle1 == duty_cycle2)
        # print('netlist', netlist)
        # print('duty_cycle', duty_cycle)

if __name__ == '__main__':
    args = get_transformer_args() 
    config_path = 'parsers/config/parser.yaml'
    config = load_and_apply_yaml_config(config_path)
    os.makedirs(config.text_data_dir, exist_ok=True)
    
    data_path = os.path.join(config.base_data_dir, config.raw_data)
    output_path = os.path.join(config.text_data_dir, config.target_data)
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate'
    # data_path = os.path.join(prefix, "data3.json")
    portion_idx = 1
    data_path = os.path.join(prefix, 'data{}.json'.format(portion_idx))
    output_path = os.path.join(prefix, "data{}_regenerate_try.json".format(portion_idx)) 


    ## running data_3.json component 3 circuit
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component3'
    portion_idx = 4
    data_path = os.path.join(prefix, 'data{}.json'.format(portion_idx))
    output_path = os.path.join(prefix, "data{}_regenerate.json".format(portion_idx)) 

    # ## running data_4.json component 4 circuit
    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/regenerate_component4'
    portion_idx = 4
    data_path = os.path.join(prefix, 'data{}.json'.format(portion_idx))
    output_path = os.path.join(prefix, "data{}_regenerate.json".format(portion_idx)) 

    prefix = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/dataset20220523/remove_redundant_circuit'
    portion_idx = 1
    data_path = os.path.join(prefix, 'dataset_345_remove_redundant.json')
    output_path = os.path.join(prefix, 'dataset_345_remove_redundant_regenerate.json')

    print('data_path', data_path)
    print('output_path', output_path)
    # parse_json_data(data_path=data_path, output_path=output_path,
    #                 select_cond=args.select_cond, 
    #                 use_log=args.use_log,
    #                 target_vout=args.target_vout)
    
    # regenerate_data_with_sim(prefix=prefix, data_path=data_path, output_path=output_path, portion_idx=portion_idx)
    # split_data_four_way(data_path=data_path)
    # combine_split_data(prefix)
    # check_sim_regenerate()
    # remove_isomorphic_circuit_main()
    # check_remove_isomorphic_circuit_main()
    # remove_isomorphic_circuit_main_all_345_component_data()

    # process_data_3()
    # extract_essensial_remove_redundant_circuit_main()
    # convert_raw_2_matrix_main()
    # test_masked_simulation()
    # check()
    # simulation_100_times()

    # random_gen_6component()

    # convert_raw_2_matrix_main_input_label()
    
    exit()
    