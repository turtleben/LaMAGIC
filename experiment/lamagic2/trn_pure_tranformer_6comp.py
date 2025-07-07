import os
import sys
dir_path = os.getcwd()
sys.path.append(dir_path)
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import torch.nn as nn
import torch

from utils.yaml_parser import load_and_apply_yaml_config
# from analog_LLM.analog_LLM import AnalogLLMBuilder
from analog_LLM.analog_transformer import AnalogTransformerBuilder
from analog_LLM.utils.data_collator import DataCollatorForT5MLM
from analog_LLM.utils.tokenizer import CustomTokenizer

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
    LLM_builder = AnalogTransformerBuilder(config)

def run_SFCI_T5tokenizer(need_to_tokenize=False, trn_data_num=500):
    config_path = 'analog_LLM/configs/pure_transformer/yml/topogen_encdec.yml'
    config = load_and_apply_yaml_config(config_path)
    config.LUT_cir_data_name = "SFCI_6comp.json"
    config.target_data = 'SFCI_6comp.json'
    config.val_set_size = 10000

    config.vocab_file = 'analog_LLM/configs/pure_transformer/dict/canonical_typeNidx_duty10first.json'
    config.tokenized_data_trn = "SFCI_6comp_trn.pickle"
    config.tokenized_data_val = "SFCI_6comp_val.pickle"
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
    config.load_pretrained = False

    config.generate = True
    config.micro_batch_size = 32
    config.reg = 1e-5

    config.prune_invalid = True
    config.normalize = False

    # config.val_custom = True
    config.num_epochs = 120
    config.eval_steps = 20

    config.data_augment = True
    config.load_pretrained = True
    config.use_duty_cycle_option_prefix = True
    config.random_causal = False
    config.dropout_rate = 0.15
    config.num_epochs = 120
    config.finetune_from_ours = True
    config.wandb_run_name = 'LaMAGIC2-345Comp-SFCI-dataaug'
    config.our_model_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name


    if trn_data_num == 2000:
        config.trn_data_num = 2000
        config.data_augment = False
        config.wandb_run_name = 'LaMAGIC2-6Comp-SFCI-dnum2000'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name + '/checkpoint-80'

    elif trn_data_num == 1000:
        config.trn_data_num = 1000
        config.data_augment = True
        config.wandb_run_name = 'LaMAGIC2-345Comp-SFCI-dnum1000-aug'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name
        LLM_builder = AnalogTransformerBuilder(config)
        LLM_builder.train()

        config.our_model_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name
        config.data_augment = False
        config.wandb_run_name = 'LaMAGIC2-6Comp-SFCI-dnum1000'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name
    
    elif trn_data_num == 500:
        config.trn_data_num = 500
        config.data_augment = False
        config.wandb_run_name = 'sLaMAGIC2-6Comp-SFCI-dnum500'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name
    
    else:
        raise ValueError("trn_data_num must be 500, 1000, or 2000")
    
    if need_to_tokenize == True:
        tokenized(config)
        return
    LLM_builder = AnalogTransformerBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True)

def run_SFM_T5tokenizer(need_to_tokenize=False, trn_data_num=500):
    config_path = 'analog_LLM/configs/pure_transformer/yml/topogen_encdec.yml'
    config = load_and_apply_yaml_config(config_path)
    config.LUT_cir_data_name = "SFM_6comp.json"
    config.target_data = 'SFM_6comp.json'
    config.tokenized_data_trn = "SFM_6comp_trn.pickle"
    config.tokenized_data_val = "SFM_6comp_val.pickle"
    config.vocab_file = 'analog_LLM/configs/pure_transformer/dict/matrix_dutyfirst.json'
    config.tokenized = True
    config.masked_method = 'full-connection' 
    config.baseline_format = 'matrix'
    config.llm = 'transformer-encoder-decoder'
    config.tokenizer = 'flanT5'
    config.val_set_size = 10000
    config.eval_steps = 80

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
    config.eval_steps = 20

    config.finetune_from_ours = True
    config.wandb_run_name = 'LaMAGIC2-345Comp-SFM'
    config.our_model_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name

    if trn_data_num == 2000:
        config.num_epochs = 120
        config.trn_data_num = 2000
        config.data_augment = True
        config.wandb_run_name = 'LaMAGIC2-6Comp-SFCI-dnum2000'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name

    elif trn_data_num == 1000:
        config.num_epochs = 120
        config.trn_data_num = 1000
        config.data_augment = True
        config.wandb_run_name = 'LaMAGIC2-6Comp-SFCI-dnum1000'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name
    
    elif trn_data_num == 500:
        config.num_epochs = 120
        config.trn_data_num = 500
        config.data_augment = False
        config.wandb_run_name = 'LaMAGIC2-6Comp-SFCI-dnum500'
        config.output_dir = '[YOUR_MODEL_SAVE_DIR]/' + config.wandb_run_name

    else:
        raise ValueError("trn_data_num must be 500, 1000, or 2000")

    if need_to_tokenize == True:
        tokenized(config)
        return
    
    LLM_builder = AnalogTransformerBuilder(config)
    LLM_builder.train() # train LLM
    LLM_builder.val(sim=True, comp6=True)

    
def plot_threshold_hist_generation():
    threshold = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    threshold_str = []
    for num in threshold:
        threshold_str.append(str(num))

    wandb_run_name = 'LaMAGIC2-6Comp-SFCI-dnum1000'
    output_dir = '[YOUR_MODEL_SAVE_DIR]/' + wandb_run_name + '/checkpoint-40'

   
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
    plt.savefig("plot_transformer/success_rate_{}.png".format(wandb_run_name), dpi=200)
    plt.close()

    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits_eff), torch.FloatTensor(scalar_labels_eff))
    print('current mse (eff ):        ', loss)
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
    The code for SFM (LaMAGIC2) that finetunes on 6-component circuit.
    First, tokenize the dataset by setting `need_to_tokenize=True` in the function call. (One time only)
    This will create the tokenized dataset files in the specified directory. You comment this line after the first run.
    Then, run the training with data amount 500 or 1000 or 2000.
    '''
    run_SFM_T5tokenizer(need_to_tokenize=True, trn_data_num=500)
    run_SFM_T5tokenizer(need_to_tokenize=False, trn_data_num=500)
    run_SFM_T5tokenizer(need_to_tokenize=False, trn_data_num=1000)
    run_SFM_T5tokenizer(need_to_tokenize=False, trn_data_num=2000)

    '''
    The code for SFCI (LaMAGIC2) that finetunes on 6-component circuit.
    '''
    run_SFCI_T5tokenizer(need_to_tokenize=True, trn_data_num=500)
    run_SFCI_T5tokenizer(need_to_tokenize=False, trn_data_num=500)
    run_SFCI_T5tokenizer(need_to_tokenize=False, trn_data_num=1000)
    run_SFCI_T5tokenizer(need_to_tokenize=False, trn_data_num=2000)

    '''
    The code for plotting the success rate under different thresholds.
    Inside the function, you  change the output_dir and wandb_run_name to the one you want to plot.
    '''
    # plot_threshold_hist_generation()