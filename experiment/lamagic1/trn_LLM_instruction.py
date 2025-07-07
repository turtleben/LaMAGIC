import os
import sys
dir_path = os.getcwd()
sys.path.append(dir_path)
from typing import List
from matplotlib import pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from scipy.stats import gaussian_kde

from utils.yaml_parser import load_and_apply_yaml_config
from analog_LLM.analog_LLM import AnalogLLMBuilder

def tokenized(config):
    """Tokenize our text dataset and save"""
    config.tokenized = False
    def find_placeholders(cfg):
        return {
            name: val
            for name, val in vars(cfg).items()
            if isinstance(val, str) and val.startswith("[YOUR_")
        }
    placeholders = find_placeholders(config)
    if placeholders:
        for name, val in placeholders.items():
            print(f"Please set the value for {name} (currently {val})")
            raise ValueError(f"Please set the value for {name} (currently {val})")
    LLM_builder = AnalogLLMBuilder(config)
    exit()

    
def train_edgeGen_naive_form():
    """Train/Val Flan-T5 for topology generation (topogen)"""
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'original'
    ## If not tokenize the dataset before, use this function. Otherwise, comment it
    
    config.tokenized = True
    config.trn_data_num = "all"
    config.finetune_from_ours = False
    
    # Need to modify to 'train' for training and 'val' for generation 
    # The model will select to load pretrained weights for training or our stored weights for validation
    config.trn_or_val = 'train'
    # Set to True when using generation in validation (if regression, set to False in validation)
    config.generate = True
    config.prune_invalid = True
    config.normalize = False
    
    config.num_epochs = 70
    # config.micro_batch_size = 32
    config.reg = 1e-5
    # The GPU device (0 or 1)
    config.LLM_device = 0
    
    # if you havn't tokenized the dataset, un-comment this two lines to tokenize the dataset
    # tokenized(config)
    # exit()

    config.wandb_run_name = 'topogen-instruction-data345new-no-augment-epoch70'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name + '/checkpoint-33400' + "/checkpoint-25600"
    
    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True) # validate LLM (in topogen, set sim=True can enable simulator to validate the generated topo)

def train_topoGen_naive_form():
    """Train/Val Flan-T5 for topology generation (topogen)"""
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'original'
    config.target_data = 'dataset_all_345_regenerate_prune_isomophic_topology.json'
    config.tokenized_data_trn = "dataset_all_345_regenerate_prune_isomophic_topology_trn.pickle"
    config.tokenized_data_val = "dataset_all_345_regenerate_prune_isomophic_topology_val.pickle"
    ## If not tokenize the dataset before, use this function. Otherwise, comment it
    
    config.tokenized = True
    config.trn_data_num = "all"
    config.finetune_from_ours = True
    
    config.trn_or_val = 'train'
    config.generate = True
    config.prune_invalid = True
    config.normalize = False
    
    config.num_epochs = 70
    config.reg = 1e-5
    config.LLM_device = 0
    
    # tokenized(config)
    # exit()
    
    config.wandb_run_name = 'topogen-instruction-data345new-no-augment-epoch70'
    config.our_model_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name + '/checkpoint-33400' + "/checkpoint-25600"
    config.wandb_run_name = 'topogen-instruction-data345new-no-augment-epoch70_topology'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name + '/checkpoint-24200'
    
    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    # LLM_builder.train() # train LLM
    LLM_builder.val(sim=True) # validate LLM (in topogen, set sim=True can enable simulator to validate the generated topo)

def train_edgeGen_canonical_form():
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'shrink_canonical'
    config.target_data = 'dataset_all_345_regenerate_prune_isomophic_shrink_canonical.json'
    config.tokenized_data_trn = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_trn.pickle"
    config.tokenized_data_val = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_val.pickle"

    config.tokenized = True
    config.trn_data_num = "all"
    config.finetune_from_ours = False
    
    config.trn_or_val = 'train'
    config.generate = True
    config.prune_invalid = True
    config.normalize = False
    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.LLM_device = 0
    
    # tokenized(config)
    # exit()
    config.wandb_run_name = 'topogen-shrink_canonical-data345new-no-augment-epoch70'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    
    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    # LLM_builder.train() # train LLM
    LLM_builder.val(sim=True) # validate LLM (in topogen, set sim=True can enable simulator to validate the generated topo)


def train_topoGen_canonical_form():
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'shrink_canonical'
    config.target_data = 'dataset_all_345_regenerate_prune_isomophic_shrink_canonical.json'
    config.tokenized_data_trn = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_topology_trn.pickle"
    config.tokenized_data_val = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_topology_val.pickle"

    config.tokenized = True
    config.trn_data_num = "all"
    config.finetune_from_ours = True
    
    config.trn_or_val = 'val'
    config.generate = True
    config.prune_invalid = True
    config.normalize = False
    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.LLM_device = 0
    
    # tokenized(config)
    # exit()
    config.wandb_run_name = 'topogen-shrink_canonical-data345new-no-augment-epoch70'
    config.our_model_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name

    config.wandb_run_name = 'topogen-shrink_canonical-data345new-no-augment-epoch70-topology'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name +'/checkpoint-14800'
    
    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    # LLM_builder.train() # train LLM
    LLM_builder.val(sim=True) # validate LLM (in topogen, set sim=True can enable simulator to validate the generated topo)

def train_edgeGen_canonical_form_one_hot_duty():
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'shrink_canonical_dutycycle'
    config.target_data = 'dataset_all_345_regenerate_prune_isomophic_shrink_canonical_dutycycle.json'
    config.tokenized_data_trn = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_dutycycle_trn.pickle"
    config.tokenized_data_val = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_dutycycle_val.pickle"

    config.tokenized = True
    config.trn_data_num = "all"
    config.finetune_from_ours = False
    
    config.trn_or_val = 'val'
    config.generate = True
    config.prune_invalid = True
    config.normalize = False
    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.LLM_device = 0
    
    # tokenized(config)
    # exit()
    config.wandb_run_name = 'topogen-shrink-canonical-dutycycle-data345new-no-augment-epoch70'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    
    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    # LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)

def train_topoGen_canonical_form_one_hot_duty():
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'shrink_canonical_dutycycle'
    config.target_data = 'dataset_all_345_regenerate_prune_isomophic_shrink_canonical_dutycycle.json'
    config.tokenized_data_trn = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_dutycycle_topology_trn.pickle"
    config.tokenized_data_val = "dataset_all_345_regenerate_prune_isomophic_new_shrink_canonical_dutycycle_topology_val.pickle"

    config.tokenized = True
    config.trn_data_num = "all"
    config.finetune_from_ours = False
    
    config.trn_or_val = 'val'
    config.generate = True
    config.prune_invalid = True
    config.normalize = False
    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.LLM_device = 0
    
    # tokenized(config)
    # exit()
    config.wandb_run_name = 'topogen-shrink-canonical-dutycycle-data345new-no-augment-epoch70'
    config.our_model_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name

    config.wandb_run_name = 'topogen-shrink-canonical-dutycycle-data345new-no-augment-epoch70-topology'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name + '/checkpoint-30600'
    
    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)

def plot_threshold_hist_generation():
    threshold = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    threshold_str = []
    for num in threshold:
        threshold_str.append(str(num))
    
    wandb_run_name = 'topogen-instruction-data345new-no-augment-epoch70'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name + '/checkpoint-33400' + "/checkpoint-25600"    
    
    wandb_run_name = 'topogen-shrink-canonical-dutycycle-data345new-no-augment-epoch70'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name

    wandb_run_name = 'topogen-shrink_canonical-data345new-no-augment-epoch70'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name

    wandb_run_name = 'topogen-shrink-canonical-dutycycle-data345new-no-augment-epoch70-topology'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name + '/checkpoint-30600'

    wandb_run_name = 'topogen-shrink_canonical-data345new-no-augment-epoch70-topology'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name +'/checkpoint-14800'

    wandb_run_name = 'topogen-instruction-data345new-no-augment-epoch70_topology'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name + '/checkpoint-24200'

    wandb_run_name = 'topogen-shrink_canonical-data345new-nonisomorphic-no-augment-epoch70'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name + '/checkpoint-11400'

    wandb_run_name = 'topogen-matrix-first-data345new-no-augment-epoch70'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name +'/checkpoint-40000'
    
    scalar_logits_vout = np.load(os.path.join(output_dir, 'scalar_logits_vout.npy'))
    scalar_labels_vout = np.load(os.path.join(output_dir, 'scalar_labels_vout.npy'))
    scalar_logits_eff = np.load(os.path.join(output_dir, 'scalar_logits_eff.npy'))
    scalar_labels_eff = np.load(os.path.join(output_dir, 'scalar_labels_eff.npy'))
    # scalar_logits = np.load(os.path.join(output_dir, 'scalar_logits_eff.npy'))
    # scalar_labels = np.load(os.path.join(output_dir, 'scalar_labels_eff.npy'))
    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_vout), torch.FloatTensor(scalar_labels_vout))
    print('current mse (vout):        ', loss)
    print(len(scalar_logits_vout))
    num = np.zeros_like(threshold)
    for i in range(len(threshold)):
        vout_bool = np.abs(scalar_logits_vout - scalar_labels_vout) <= threshold[i]
        eff_bool = (scalar_labels_eff - scalar_logits_eff) <= threshold[i]
        # print(np.sum(np.multiply( vout_bool, eff_bool  )), np.sum(vout_bool), np.sum(eff_bool))
        num[i] = np.round(np.sum(np.multiply( vout_bool, eff_bool  )) / len(scalar_labels_vout), 2)
        # num[i] = np.sum(np.multiply( vout_bool, eff_bool ))
        # num[i] = np.round(np.sum( scalar_labels - scalar_logits <= threshold[i]) / len(scalar_labels), 2)
    print(num)
    print(threshold_str)
    rect = plt.bar(threshold_str, num)
    plt.bar_label(rect, padding=3)
    plt.xlabel('threshold')
    plt.ylabel('number of samples')
    plt.title('Success rates under differet thresholds')
    plt.savefig("plot/success_rate_topogen-matrix-first-data345new-no-augment.png", dpi=200)
    plt.close()

    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_eff), torch.FloatTensor(scalar_labels_eff))
    xy = np.vstack([scalar_labels_eff, scalar_logits_eff])
    z = gaussian_kde(xy)(xy)
    plt.scatter(scalar_labels_eff, scalar_logits_eff, s=10, c=z)
    plt.plot(scalar_labels_eff, scalar_labels_eff, linewidth=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Efficiency inputs')
    plt.ylabel('Efficiency of generated topologies')
    plt.title('Efficiency, mse: ' + str(round(loss.item(), 4)))
    plt.savefig("plot/eff_topogen-matrix-first-data345new-no-augment.png", dpi=200)
    plt.close()

    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_vout), torch.FloatTensor(scalar_labels_vout))
    xy = np.vstack([scalar_labels_vout, scalar_logits_vout])
    z = gaussian_kde(xy)(xy)
    plt.scatter(scalar_labels_vout, scalar_logits_vout, s=10, c=z)
    plt.plot(scalar_labels_vout, scalar_labels_vout, linewidth=0.5)
    plt.xlim([np.min(scalar_labels_vout), np.max(scalar_labels_vout)])
    plt.ylim([np.min(scalar_labels_vout), np.max(scalar_labels_vout)])
    plt.xlim([-1, np.max(scalar_labels_vout)])
    plt.ylim([-1, np.max(scalar_labels_vout)])
    plt.xlabel('Voltage conversion ratio inputs')
    plt.ylabel('Voltage conversion ratio of generated topologies')
    plt.title('Voltage conversion ratio, mse: ' + str(round(loss.item(), 4)))
    plt.savefig("plot/vout_topogen-matrix-first-data345new-no-augment.png", dpi=200)

if __name__ == "__main__":
    train_edgeGen_canonical_form()
    
