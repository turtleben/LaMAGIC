import os
import sys
dir_path = os.getcwd()
sys.path.append(dir_path)
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch.nn as nn
import torch
import copy

from utils.yaml_parser import load_and_apply_yaml_config
from analog_LLM.analog_LLM import AnalogLLMBuilder
from analog_LLM.utils.data_collator import DataCollatorForT5MLM

def tokenized(config):
    """Tokenize our text dataset and save"""
    config.tokenized = False
    LLM_builder = AnalogLLMBuilder(config)


def trn_edgeGen_vertex_permutation():
    """Train/Val Flan-T5 for topology generation (topogen)"""
    config_path = 'analog_LLM/configs/masked/topogen_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.masked_ratio = 0.5
    
    # "train" or "val"
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = False
    config.generate = True    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5
    
    config.prune_invalid = True
    config.data_augment = True
    config.num_epochs = 120
    config.wandb_run_name = 'LaMAGIC-345component-edgeGen-float-input-matrix-form-dataaug'
    config.output_dir = config.wandb_run_name

    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM``
    LLM_builder.val(sim=True)

def trn_edgeGen_vertex_permutation_then_no_permute():
    config_path = 'analog_LLM/configs/masked/topogen_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.masked_ratio = 0.5
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = True
    config.generate = True    
    config.num_epochs = 30
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.prune_invalid = True
    config.normalize = False
    config.data_augment = False
    
    config.wandb_run_name = 'LaMAGIC-345component-edgeGen-float-input-matrix-form-dataaug'
    config.our_model_dir = config.wandb_run_name
    config.wandb_run_name = 'LaMAGIC-345component-edgeGen-float-input-matrix-form-dataaug-noaug'
    config.output_dir =    config.wandb_run_name

    # The controller for LLM training
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)


def trn_edgeGen_then_topologyGen():
    config_path = 'analog_LLM/configs/masked/topogen_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.tokenized = True
    config.masked_method = 'full-graph'
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = True
    config.generate = True    
    
    config.num_epochs = 40
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.prune_invalid = True

    config.wandb_run_name = 'LaMAGIC-345component-edgeGen-float-input-matrix-form-dataaug-noaug'
    config.our_model_dir =    config.wandb_run_name
    
    config.eval_steps = 20
    config.wandb_run_name = 'LaMAGIC-345component-topoGen-float-input-matrix-form'
    config.output_dir =    config.wandb_run_name
    
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)

def plot_threshold_hist_generation():
    threshold = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    threshold_str = []
    for num in threshold:
        threshold_str.append(str(num))

    wandb_run_name = 'LaMAGIC-345component-topoGen-float-input-matrix-form'
    output_dir =    wandb_run_name
    
    
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
    plt.savefig("plot/success_rate_data345new-noisormorphic-connection-dataaug-noaug-epoch30.png", dpi=200)
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
    plt.savefig("plot/eff_data345new-noisormorphic-connection-dataaug-noaug-epoch30.png", dpi=200)
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
    plt.savefig("plot/vout_data345new-noisormorphic-connection-dataaug-noaug-epoch30.png", dpi=200)

if __name__ == "__main__":
    # run()
    trn_edgeGen_vertex_permutation()
    trn_edgeGen_vertex_permutation_then_no_permute()