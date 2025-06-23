import os
import sys
dir_path = os.getcwd()
sys.path.append(dir_path)
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch.nn as nn
import torch
import json
import copy

from utils.yaml_parser import load_and_apply_yaml_config
# from analog_LLM.analog_LLM import AnalogLLMBuilder
from analog_LLM.analog_transformer import AnalogTransformerBuilder
from analog_LLM.utils.data_collator import DataCollatorForT5MLM
from analog_LLM.utils.tokenizer import CustomTokenizer

def tokenized(config):
    """Tokenize our text dataset and save"""
    config.tokenized = False
    LLM_builder = AnalogTransformerBuilder(config)

def run_topogen_matrix_dutyfirst_T5tokenizer_dataaug(need_to_tokenize=False):
    config_path = 'analog_LLM/configs/pure_transformer/yml/topogen_encdec.yml'
    config = load_and_apply_yaml_config(config_path)
    config.target_data = 'dataset_all_345_matrix_dutycycle_first.json'
    config.tokenized_data_trn = "dataset_345_matrix_dutycycle_first_tokenizerflant5_trn.pickle"
    config.tokenized_data_val = "dataset_345_matrix_dutycycle_first_tokenizerflant5_val.pickle"
    config.vocab_file = 'analog_LLM/configs/pure_transformer/dict/matrix_dutyfirst.json'
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.baseline_format = 'matrix'
    config.llm = "flan-t5"
    config.llm = 'transformer-encoder-decoder'
    config.tokenizer = 'flanT5'

    # set to 'train' or 'val'
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = False
    config.load_pretrained = True

    config.generate = True
    config.micro_batch_size = 32
    config.reg = 1e-5

    config.prune_invalid = True
    config.normalize = False
    config.num_epochs = 120
    config.eval_steps = 400

    config.data_augment = True
    config.wandb_run_name = 'matrix-loadT5andTokenizer-data345new-connection-aug-epoch120'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + config.wandb_run_name

    # If it is the first time running, run the following line to create and save the tokenized dataset
    if need_to_tokenize == True:
        tokenized(config)
        return

    LLM_builder = AnalogTransformerBuilder(config)
    LLM_builder.train() # train LLM
    # LLM_builder.val(sim=True)

def run_topogen_matrix_dutyfirst_T5tokenizer_noaug(need_to_tokenize=False):
    config_path = 'analog_LLM/configs/pure_transformer/yml/topogen_encdec.yml'
    config = load_and_apply_yaml_config(config_path)
    config.target_data = 'dataset_all_345_matrix_dutycycle_first.json'
    config.tokenized_data_trn = "dataset_345_matrix_dutycycle_first_tokenizerflant5_trn.pickle"
    config.tokenized_data_val = "dataset_345_matrix_dutycycle_first_tokenizerflant5_val.pickle"
    config.vocab_file = 'analog_LLM/configs/pure_transformer/dict/matrix_dutyfirst.json'
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.baseline_format = 'matrix'
    config.llm = "flan-t5"
    config.llm = 'transformer-encoder-decoder'
    config.tokenizer = 'flanT5'

    # set to 'train' or 'val'
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.finetune_from_ours = False
    config.load_pretrained = True

    config.generate = True
    config.micro_batch_size = 32
    config.reg = 1e-5

    config.prune_invalid = True
    config.normalize = False
    config.num_epochs = 120
    config.eval_steps = 400

    config.wandb_run_name = 'matrix-loadT5andTokenizer-data345new-connection-aug-epoch120'
    config.our_model_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + config.wandb_run_name
    config.data_augment = False
    config.finetune_from_ours = True
    config.wandb_run_name = 'matrix-loadT5andTokenizer-data345new-connection-aug-noaug-epoch120'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + config.wandb_run_name

    LLM_builder = AnalogTransformerBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)

def run_topogen_shrink_canonical_typeNidx_dutyfirst_T5tokenizer_dataaug(need_to_tokenize=False):
    config_path = 'analog_LLM/configs/pure_transformer/yml/topogen_encdec.yml'
    config = load_and_apply_yaml_config(config_path)
    config.target_data = 'dataset_345_shrink_canonical_typeNidx_dutycycle_first.json'
    config.vocab_file = 'analog_LLM/configs/pure_transformer/dict/canonical_typeNidx_duty10first.json'
    config.tokenized_data_trn = "dataset_345_shrink_canonical_typeNidx_dutycycle_first_tokenizerflant5_trn.pickle"
    config.tokenized_data_val = "dataset_345_shrink_canonical_typeNidx_dutycycle_first_tokenizerflant5_val.pickle"
    config.use_duty_cycle_option_prefix = True
    config.typeNidx = True
    config.tokenized = True
    config.duty10 = False
    config.tokenizer = 'flanT5'

    config.masked_method = 'full-connection' 
    config.baseline_format = 'shrink_canonical'
    config.vocab_size = 34
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.generate = True
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.prune_invalid = True
    config.normalize = False
    config.num_epochs = 120
    config.eval_steps = 400

    # val in 2413 tmux 6, need to change data_collator 
    # current best formulation
    config.data_augment = True
    config.load_pretrained = False
    config.use_duty_cycle_option_prefix = True
    config.random_causal = False
    config.dropout_rate = 0.15
    config.wandb_run_name = 'shrink_canonical_typeNidx_loadT5andTokenizer_useduty-data345new-connection-aug-epoch120'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + config.wandb_run_name

    # If it is the first time running, run the following line to create and save the tokenized dataset
    if need_to_tokenize == True:
        tokenized(config)
        return

    LLM_builder = AnalogTransformerBuilder(config)
    LLM_builder.train() # train LLM
    # LLM_builder.val(sim=True)

def run_topogen_shrink_canonical_typeNidx_dutyfirst_T5tokenizer_noaug():
    config_path = 'analog_LLM/configs/pure_transformer/yml/topogen_encdec.yml'
    config = load_and_apply_yaml_config(config_path)
    config.target_data = 'dataset_345_shrink_canonical_typeNidx_dutycycle_first.json'
    config.vocab_file = 'analog_LLM/configs/pure_transformer/dict/canonical_typeNidx_duty10first.json'
    config.tokenized_data_trn = "dataset_345_shrink_canonical_typeNidx_dutycycle_first_tokenizerflant5_trn.pickle"
    config.tokenized_data_val = "dataset_345_shrink_canonical_typeNidx_dutycycle_first_tokenizerflant5_val.pickle"
    config.use_duty_cycle_option_prefix = True
    config.typeNidx = True
    config.tokenized = True
    config.duty10 = False
    config.tokenizer = 'flanT5'

    config.masked_method = 'full-connection' 
    config.baseline_format = 'shrink_canonical'
    config.vocab_size = 34
    
    config.trn_or_val = 'train'
    config.trn_data_num = "all"
    config.LLM_device = 0
    config.generate = True
    config.micro_batch_size = 32
    config.reg = 1e-5
    config.prune_invalid = True
    config.normalize = False
    config.num_epochs = 120
    config.eval_steps = 400

    # current best formulation
    config.load_pretrained = False
    config.use_duty_cycle_option_prefix = True
    config.random_causal = False
    config.wandb_run_name = 'shrink_canonical_typeNidx_loadT5andTokenizer_useduty-data345new-connection-aug-epoch120'
    config.our_model_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + config.wandb_run_name
    config.data_augment = False
    config.dropout_rate = 0.1
    config.finetune_from_ours = True
    config.wandb_run_name = 'shrink_canonical_typeNidx_loadT5andTokenizer_useduty-data345new-connection-aug-noaug-epoch120'
    config.output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + config.wandb_run_name

    LLM_builder = AnalogTransformerBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)
    
def plot_threshold_hist_generation():
    threshold = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    threshold_str = []
    for num in threshold:
        threshold_str.append(str(num))

    wandb_run_name = 'matrix-loadT5andTokenizer-data345new-connection-aug-noaug-epoch120'
    output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + wandb_run_name

    # wandb_run_name = 'shrink_canonical_typeNidx_loadT5andTokenizer_useduty-data345new-connection-aug-noaug-epoch120'
    # output_dir = '/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/LLM_models/encoder_decoder/' + wandb_run_name


    data_generated = json.load(open(os.path.join(output_dir, 'data_generated.json'), 'r'))
    print('data_generated:', len(data_generated))
    
    
    scalar_logits_vout = np.load(os.path.join(output_dir, 'scalar_logits_vout.npy'))
    scalar_labels_vout = np.load(os.path.join(output_dir, 'scalar_labels_vout.npy'))
    scalar_logits_eff = np.load(os.path.join(output_dir, 'scalar_logits_eff.npy'))
    scalar_labels_eff = np.load(os.path.join(output_dir, 'scalar_labels_eff.npy'))

    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_vout), torch.FloatTensor(scalar_labels_vout))
    print('current mse (vout):        ', loss)
    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_eff), torch.FloatTensor(scalar_labels_eff))
    print('current mse (eff):        ', loss)
    print(len(scalar_labels_eff))
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
    plt.savefig("plot_transformer/success_rate_{}.png".format(wandb_run_name), dpi=200)
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
    plt.savefig("plot_transformer/eff_{}.png".format(wandb_run_name), dpi=200)
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
    plt.savefig("plot_transformer/vout_{}.png".format(wandb_run_name), dpi=200)




if __name__ == "__main__":

    '''
    The code for SFM (LaMAGIC2) training with T5 tokenizer and data augmentation
    First, tokenize the dataset by setting `need_to_tokenize=True` in the function call. (One time only)
    This will create the tokenized dataset files in the specified directory. You comment this line after the first run.
    Then, run the training with data augmentation by setting `need_to_tokenize=False`.
    Next, run the training and validation without data augmentation. 
    It loads the model with data augmentation and finetunes it without data augmentation.
    '''
    run_topogen_matrix_dutyfirst_T5tokenizer_dataaug(need_to_tokenize=True)
    run_topogen_matrix_dutyfirst_T5tokenizer_dataaug(need_to_tokenize=False)
    run_topogen_matrix_dutyfirst_T5tokenizer_noaug()

    '''
    The code for SFCI (LaMAGIC2) training with T5 tokenizer and data augmentation
    '''
    run_topogen_shrink_canonical_typeNidx_dutyfirst_T5tokenizer_dataaug(need_to_tokenize=True)
    run_topogen_shrink_canonical_typeNidx_dutyfirst_T5tokenizer_dataaug(need_to_tokenize=False)
    run_topogen_shrink_canonical_typeNidx_dutyfirst_T5tokenizer_noaug()

    '''
    The code for plotting the success rate under different thresholds
    Inside this function, you should change the output_dir (dir that saves model params) and wandb_run_name to the one you want to plot.
    '''
    plot_threshold_hist_generation()