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


def trn_PM_6_comp(need_tokenized=False, data_num=500):
    config_path = 'analog_LLM/configs/masked/topogen_flanT5_6_component.yml'
    config = load_and_apply_yaml_config(config_path)
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.masked_ratio = 0.5
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = True
    config.generate = True    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5

    config.prune_invalid = True
    config.normalize = False

    config.llm = "flan-t5-baseline" # This controls whether use float input
    config.eval_steps = 40
    config.wandb_run_name = 'LaMAGIC-345component-pure-text-matrix-form-dataaug-noaug'
    config.our_model_dir = config.wandb_run_name
    config.data_augment = True


    if data_num == 500:
        config.trn_data_num = 500
        config.wandb_run_name = 'flanT5-maskedgen-nofloatembed-data6-aug-from-connection-dataaug-noaug-epoch120-dnum500'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    elif data_num == 1000:
        config.trn_data_num = 1000
        config.wandb_run_name = 'flanT5-maskedgen-nofloatembed-data6-aug-from-connection-dataaug-noaug-epoch120-dnum1000'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    elif data_num == 2000:
        config.trn_data_num = 2000
        config.wandb_run_name = 'flanT5-maskedgen-nofloatembed-data6-aug-from-connection-dataaug-noaug-epoch120-dnum2000'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name

    if need_tokenized:
        tokenized(config)
        return

    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True, comp6=True)

def trn_FM_6_comp(need_tokenized=False, data_num=500):
    config_path = 'analog_LLM/configs/masked/topogen_flanT5_6_component.yml'
    config = load_and_apply_yaml_config(config_path)
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.masked_ratio = 0.5
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = True
    config.generate = True    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5

    config.prune_invalid = True
    config.normalize = False
    config.data_augment = True

    config.wandb_run_name = 'LaMAGIC-345component-edgeGen-float-input-matrix-form-dataaug-noaug'
    config.our_model_dir  = config.wandb_run_name

    if data_num == 500:
        config.trn_data_num = 500
        config.wandb_run_name = 'flanT5-maskedgen-data6-aug-from-connection-dataaug-noaug-epoch120-dnum500'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    elif data_num == 1000:
        config.wandb_run_name = 'flanT5-maskedgen-data6-aug-from-connection-dataaug-noaug-epoch120-dnum1000'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    elif data_num == 2000:
        config.wandb_run_name = 'flanT5-maskedgen-data6-aug-from-connection-dataaug-noaug-epoch120-dnum2000'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    else:
        raise ValueError("data_num must be 500, 1000, or 2000")
    if need_tokenized:
        tokenized(config)
        return
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True, comp6=True)

def trn_CF_6_comp(need_tokenized=False, data_num=500):
    config_path = 'analog_LLM/configs/instruction/topogen_instruction_flanT5.yml'
    config = load_and_apply_yaml_config(config_path)
    config.LUT_cir_data_name = "dataset_6_regenerate_LUT_for_eval.json"
    config.llm = 'flan-t5-baseline'
    config.baseline_format = 'shrink_canonical'
    config.target_data = 'dataset_6_regenerate_shrink_canonical.json'
    config.tokenized_data_trn = "dataset_6_regenerate_shrink_canonical_trn.pickle"
    config.tokenized_data_val = "dataset_6_regenerate_shrink_canonical_val.pickle"

    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.masked_ratio = 0.5
    config.val_set_size = 10000
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = True
    config.generate = True    
    config.num_epochs = 70
    config.micro_batch_size = 32
    config.reg = 1e-5

    config.prune_invalid = True
    config.normalize = False
    # config.val_custom = True
    config.data_augment = False
    config.eval_steps = 40

    config.wandb_run_name = 'topogen-shrink_canonical-data345new-no-augment-epoch70'
    config.our_model_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name

    if data_num == 500:
        config.trn_data_num = 500
        config.wandb_run_name = 'topogen-shrink_canonical-data6-no-augment-epoch70-dnum500'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    elif data_num == 1000:
        config.trn_data_num = 1000
        config.wandb_run_name = 'topogen-shrink_canonical-data6-no-augment-epoch70-dnum1000'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name
    elif data_num == 2000:
        config.trn_data_num = 2000
        config.wandb_run_name = 'topogen-shrink_canonical-data6-no-augment-epoch70-dnum2000'
        config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + config.wandb_run_name

    if need_tokenized:
        tokenized(config)
        return
    
    LLM_builder = AnalogLLMBuilder(config)
    LLM_builder.train() # trsain LLM
    LLM_builder.val(sim=True, comp6=True)


def plot_threshold_hist_generation():
    threshold = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    threshold_str = []
    for num in threshold:
        threshold_str.append(str(num))

    wandb_run_name = 'topogen-shrink_canonical-data6-no-augment-epoch70-dnum2000'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/analog_LLM_model_flanT5/' + wandb_run_name

    scalar_logits_vout = np.load(os.path.join(output_dir, 'scalar_logits_vout.npy'))
    scalar_labels_vout = np.load(os.path.join(output_dir, 'scalar_labels_vout.npy'))
    scalar_logits_eff = np.load(os.path.join(output_dir, 'scalar_logits_eff.npy'))
    scalar_labels_eff = np.load(os.path.join(output_dir, 'scalar_labels_eff.npy'))
    # scalar_logits = np.load(os.path.join(output_dir, 'scalar_logits_eff.npy'))
    # scalar_labels = np.load(os.path.join(output_dir, 'scalar_labels_eff.npy'))
    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_vout), torch.FloatTensor(scalar_labels_vout))
    print('current mse (vout):        ', loss)
    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_eff), torch.FloatTensor(scalar_labels_eff))
    print('current mse (eff):        ', loss)
    print(len(scalar_logits_vout))
    num = np.zeros_like(threshold)
    for i in range(len(threshold)):
        vout_bool = np.abs(scalar_logits_vout - scalar_labels_vout) <= threshold[i]
        eff_bool = (scalar_labels_eff - scalar_logits_eff) <= threshold[i]
        # print(np.sum(np.multiply( vout_bool, eff_bool  )), np.sum(vout_bool), np.sum(eff_bool))
        # num[i] = np.round(np.sum(np.multiply( vout_bool, eff_bool  )) / len(scalar_labels_vout), 2)
        num[i] = np.round(np.sum(np.multiply( vout_bool, eff_bool  )) / 8789, 2)
        # num[i] = np.sum(np.multiply( vout_bool, eff_bool ))
        # num[i] = np.round(np.sum( scalar_labels - scalar_logits <= threshold[i]) / len(scalar_labels), 2)
    print(num)
    print(threshold_str)
    rect = plt.bar(threshold_str, num)
    plt.bar_label(rect, padding=3)
    plt.xlabel('threshold')
    plt.ylabel('number of samples')
    plt.title('Success rates under differet thresholds')
    plt.savefig("plot/success_rate-{}.png".format(wandb_run_name), dpi=200)
    plt.close()

    xy = np.vstack([scalar_labels_eff, scalar_logits_eff])
    z = gaussian_kde(xy)(xy)
    plt.scatter(scalar_labels_eff, scalar_logits_eff, s=10, c=z)
    plt.plot(scalar_labels_eff, scalar_labels_eff, linewidth=0.5)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Efficiency inputs')
    plt.ylabel('Efficiency of generated topologies')
    plt.title('Efficiency, mse: ' + str(round(loss.item(), 4)))
    plt.savefig("plot/eff_{}.png".format(wandb_run_name), dpi=200)
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
    plt.savefig("plot/vout_{}.png".format(wandb_run_name), dpi=200)

if __name__ == "__main__":
    '''
    The code to train PM in 6-component circuits
    First, tokenize the dataset by setting `need_to_tokenize=True` in the function call. (One time only)
    This will create the tokenized dataset files in the specified directory. You comment this line after the first run.
    Then, run the training by setting `need_to_tokenize=False` and varied data numbers 500, 1000, or 2000.
    '''
    trn_PM_6_comp(need_tokenized=True,  data_num=500)
    trn_PM_6_comp(need_tokenized=False, data_num=500)
    trn_PM_6_comp(need_tokenized=False, data_num=1000)
    trn_PM_6_comp(need_tokenized=False, data_num=2000)

    '''
    The code to train FM in 6-component circuits
    '''
    trn_FM_6_comp(need_tokenized=True,  data_num=500)
    trn_FM_6_comp(need_tokenized=False, data_num=500)
    trn_FM_6_comp(need_tokenized=False, data_num=1000)
    trn_FM_6_comp(need_tokenized=False, data_num=2000)

    '''
    The code to train CF in 6-component circuits
    '''
    trn_CF_6_comp(need_tokenized=True,  data_num=500)
    trn_CF_6_comp(need_tokenized=False, data_num=500)
    trn_CF_6_comp(need_tokenized=False, data_num=1000)
    trn_CF_6_comp(need_tokenized=False, data_num=2000)

    '''
    The code to plot the threshold histogram of the generation results
    Inside the function, change the wandb_run_name and output_dir to the one you want to plot
    '''
    plot_threshold_hist_generation()
