import os
import sys
import pickle
from tqdm import tqdm
# from datasets import load_dataset
import transformers
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import json
# from peft import PeftModel

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaForSequenceClassification, LlamaTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from analog_LLM.utils.params import *
from analog_LLM.utils.dataset import SupervisedDataset
from parsers.simulation import sim_generation_output, read_LLM_ouput, read_masked_LLM_output, sim_masked_generation_output, read_transformer_output_mask
from parsers.simulation import convert_netlist_2_graph, read_LLM_output_shrink_canonical, read_LLM_output_shrink_canonical_dutycycle, read_transformer_output_shrink_canonical, read_transformer_output_shrink_canonical_output_no_type, read_transformer_matrix_half
from parsers.util import jdump
from analog_LLM.utils.data_collator import DataCollatorForT5MLM, compute_input_and_target_lengths
from analog_LLM.utils.dataset import DataCollatorForSupervisedDataset


# def load_data(args):
#     d_path = os.path.join(args.text_data_dir, args.target_data)
#     print('#### data path: ', d_path)
#     data = load_dataset("json", data_files=d_path)
    
#     return data

def load_tokenized_data(tokenized_data_dir, tokenized_data, split_trn_val=False, val_set_size=0.1):
    d_path = os.path.join(tokenized_data_dir, tokenized_data)
    # print('#### data path: ', d_path)
    with open(d_path, 'rb') as f:
        data_dict = pickle.load(f)
    # data_dict["labels"] = (data_dict["labels"] -  torch.min(data_dict["labels"])) / torch.max(data_dict["labels"])
    if split_trn_val:
        X_trn, X_test, y_trn, y_test = train_test_split(data_dict["input_ids"],  \
                        data_dict["labels"], test_size=val_set_size, random_state=42)
        return X_trn, X_test, y_trn, y_test
    else:
        return data_dict

def random_split_trn_val(args, data, val_set_size):
    if val_set_size > 1:
        val_set_size = float(val_set_size) / float(len(data))
        print('#### val_set_size: ', val_set_size)
    data_trn, data_val = train_test_split(data, test_size=val_set_size, random_state=42)
    return data_trn, data_val

    # node_tokens = set()
    # type_str = ['Sa', 'Sb', 'C', 'L']
    # for device in type_str:
    #     for i in range(5):
    #         device_str = device + str(i)
    #         node_tokens.add(device_str)
    # node_tokens.add('IN')
    # node_tokens.add('OUT')
    # node_tokens.add('0')
    # trn_graphs = []
    # data_ids = []
    # for idx, datum in enumerate(tqdm(data)):
    #     # netlist, duty_cycle = read_LLM_output_shrink_canonical(datum['output'])
    #     netlist, duty_cycle = read_masked_LLM_output(datum['circuit_str'], args.order)
    #     graph = convert_netlist_2_graph(node_tokens, netlist)
    #     if len(trn_graphs) == 0:
    #         trn_graphs.append([graph])
    #         data_ids.append([idx])
    #     else:
    #         for i, trn_graph in enumerate(trn_graphs):
    #             if nx.vf2pp_is_isomorphic(trn_graph[0], graph, node_label='type'):
    #                 trn_graphs[i].append(graph)
    #                 data_ids[i].append(idx)
    #                 break
    #             elif i == len(trn_graphs)-1:
    #                 trn_graphs.append([graph])
    #                 data_ids.append([idx])
    # # help me select 10% of list in data_ids
    # data_trn = []
    # data_val = []
    # data_ids_trn, data_ids_val = train_test_split(data_ids, test_size=val_set_size, random_state=42)
    # for data_id in data_ids_trn:
    #     for idx in data_id:
    #         data_trn.append(data[idx])
    # for data_id in data_ids_val:
    #     for idx in data_id:
    #         data_val.append(data[idx])
    # return data_trn, data_val


def llm_model(args):
    if args.task == 'causal':
        return LlamaForCausalLM
    elif args.task == 'regression':
        return LlamaForSequenceClassification
    

def tokenize(args, tokenizer, prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    cutoff_len = args.cutoff_len
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(args, prompter, data_point):
    # convert each data point into single string
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)
    if not args.train_on_inputs:
        user_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"]
        )
        tokenized_user_prompt = tokenize(
            user_prompt, add_eos_token=args.add_eos_token
        )
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        if args.add_eos_token:
            user_prompt_len -= 1

        tokenized_full_prompt["labels"] = [
            -100
        ] * user_prompt_len + tokenized_full_prompt["labels"][
            user_prompt_len:
        ]  # could be sped up, probably
    return tokenized_full_prompt

def finetune(args, model, tokenizer, data_collator, dset_trn, dset_val):
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    _, args_trn = generate_config_param(args)
    if not args.ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dset_trn,
        eval_dataset=dset_val,
        args=args_trn,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    # with torch.autocast("cuda"):
    trainer.train()

    model.save_pretrained(args.output_dir)
    
    

def finetune_lora(args, model, tokenizer, data_collator, dset_trn, dset_val):
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    # if args.val_set_size > 0:
    #     val_d_num = int(len(data["train"])*args.val_set_size)
    #     args.val_set_size = val_d_num
    #     train_val = data["train"].train_test_split(
    #         test_size=val_d_num, shuffle=True, seed=42
    #     )
    #     # train_data = (
    #     #     train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    #     # )
    #     # val_data = (
    #     #     train_val["test"].shuffle().map(generate_and_tokenize_prompt)
    #     # )
    #     train_data = train_val["train"].shuffle()
    #     val_data = train_val["test"].shuffle()
    # else:
    #     print("Warning! There is no validation set")
    #     train_data = data["train"].shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    # print(model.score.weight.size(), model.score.weight)
    model = prepare_model_for_int8_training(model)
    
    config_lora, args_trn = generate_config_param(args)
    
    model = get_peft_model(model, config_lora)
    model.print_trainable_parameters()
    # print(model.base_model.model.score.weight.size(), model.base_model.model.score.weight)
    
    resume_from_checkpoint = False
    if args.resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            args.resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                args.resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")
    
    if not args.ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dset_trn,
        eval_dataset=dset_val,
        args=args_trn,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    
    # with torch.autocast("cuda"):
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(args.output_dir)
    # print(model.base_model.model.score.size(), model.score)

def val(args, model, tokenizer, dset_trn, dset_val, data_collator, cir_data, get_mse=False, sim=False):
    def save_logits(scalar_logits, scalar_labels, metrics='vout'):
        with open(os.path.join(args.output_dir, 'scalar_logits_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_logits)
        with open(os.path.join(args.output_dir, 'scalar_labels_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_labels)
        
        # scalar_logits = np.load(os.path.join(args.output_dir, 'scalar_logits_{}_dosample.npy'.format(metrics)))
        # scalar_labels = np.load(os.path.join(args.output_dir, 'scalar_labels_{}_dosample.npy'.format(metrics)))
        # plt.scatter(scalar_labels, scalar_logits)
        # plt.xlabel('Voltage conversion ratio labels')
        # if get_mse:
        #     plt.ylabel('Voltage conversion ratio predictions')
        # else:
        #     plt.ylabel('Voltage conversion ratio generations')
        # plt.savefig(args.output_dir + "/logit_label_{}.png".format(metrics), dpi=300)
        # plt.close()
    temperature=1
    top_p=0.90
    top_k=5
    num_beams=4
    max_new_tokens=256
    stream_output=False
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    generation_param = {
        # "temperature": temperature,
        # 'top_p': top_p,
        # 'top_k': top_k,
        # 'num_beams': num_beams,
        # 'do_sample': True,
        'max_new_tokens': max_new_tokens,
        # 'repetition_penalty': 1.5,
        'output_scores': True
    }
#     Input:  VIN VOUT GND Sa0 Sb0 Sb1 Sb2 C0 <sep> </s>
# Output: <pad> <duty_0.9> <sep> VIN Sb2 C0 , VOUT Sa0 Sb2 C0 , GND Sb0 Sb1 , Sa0 Sb0 Sb1 <sep> </s>
# Label:  <duty_0.7> <sep> VIN Sb0 Sb1 Sb2 C0 , VOUT Sb0 Sb1 , GND Sa0 C0 , Sa0 Sb2 <sep> </s>
    # voltage label:  0.9839336276054382 output:  0.9756038026747282
    # eff     label:  0.9590846300125122 output:  0.9435786473544071
    wrong_graph_num = 0
    invalid_graph_num = 0
    new_graph_num = 0
    if args.finetune_method == 'lora':
        model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()

    trn_graphs, trn_duty_cycles, trn_cir_strs, trn_effs, trn_vouts, node_tokens = report_trn_cir_data(args, cir_data)
    
    scalar_logits = []
    scalar_labels = []
    eff_logits = []
    eff_labels = []
    data_generated = []
    expanded_inputs_length, targets_length = compute_input_and_target_lengths(
        inputs_length=args.cutoff_len,
        noise_density=0.3,
        mean_noise_span_length=3,
    )
    # padding_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, task='conditionalGen')
    total_loss = []
    loader_val = DataLoader(dset_val, batch_size=1,
                shuffle=False, num_workers=0, pin_memory=True, collate_fn=data_collator)
    for idx, data in enumerate(tqdm(loader_val)):
        # data = padding_collator(data)
        input_ids = data["input_ids"].to('cuda')
        labels = data["labels"].to('cuda')

        # if idx > 10:
        #     break
        with torch.no_grad():
            if args.llm == 'transformer-encoder-decoder':
                vout = data['vout'].to('cuda')
                eff = data['eff'].to('cuda')

                if args.use_duty_cycle_option_prefix == False:
                    generation_output = model.generate(
                        input_ids=input_ids, d_cycle_option=None, vout=vout, eff=eff,
                        generation_config=generation_config, **generation_param
                    )
                else:
                    d_cycle_option = data['d_cycle_option'].to('cuda')
                    # print(vout, eff)
                    generation_output = model.generate(
                        input_ids=input_ids, d_cycle_option=d_cycle_option, vout=vout, eff=eff,
                        generation_config=generation_config, **generation_param
                    )
                # generation_output = model(input_ids=input_ids,  labels=labels, d_cycle_option=d_cycle_option, vout=vout, eff=eff)
                # # print(generation_output.loss)
                # total_loss.append(generation_output.loss.item())
            else:
                vout = data['vout'].to('cuda')
                eff = data['eff'].to('cuda')
                d_cycle_option = data['d_cycle_option'].to('cuda')
                # print(vout, eff, d_cycle_option)
                # generation_output = model.generate(
                #     input_ids=input_ids, d_cycle_option=d_cycle_option, vout=vout, eff=eff,
                #         generation_config=generation_config, **generation_param
                #     # input_ids=input_ids,
                #     # generation_config=generation_config,
                #     # **generation_param
                # )
                generation_outp = model(input_ids=input_ids,  labels=labels, d_cycle_option=d_cycle_option, vout=vout, eff=eff)
                total_loss.append(generation_outp.loss.item())
                # print(generation_outp.loss)
        # continue
                
        d_dict = {}
        inputs = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
        label = tokenizer.decode(labels[0], skip_special_tokens=False)
        d_dict["input"] = inputs
        d_dict["output"] = output
        d_dict["label"] = label
        
        # output = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        # label = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # # print("Current data id: ", idx)
        # print("Input: ", inputs)
        # print("Output:", output)
        # print("Label: ", label)
        # input()
        
        if get_mse:
            # print('Start calculate mse ...')
            out_strings = output.split()
            label_strings = label.split()
            # print(out_strings[-3], out_strings[-2], out_strings[-1])
            # if out_strings[-2] == 'is' and out_strings[-3] == 'ratio':
            try:
                logit = float(out_strings[-1][:len(out_strings[-1])-1])
                label = float(label_strings[-1][:len(label_strings[-1])-1])
                if label >= 3 or label <= -3:
                    continue
                # print(logit, label)
                scalar_logits.append(logit)
                scalar_labels.append(label)
            except:
                continue
            if get_mse and idx % 500 == 0:
                loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
                print('current mse: ', loss)
            
        if sim:
            try:
                path = os.path.join(args.output_dir, 'sim.cki')
                if args.baseline_format == "original":
                    netlist, duty_cycle = read_LLM_ouput(output)
                    input_strings = inputs.split()
                    label_eff = float(input_strings[-1][:len(input_strings[-1])-1])
                    label_vout = float(input_strings[-6][:len(input_strings[-1])-1])
                elif args.baseline_format == "shrink_canonical":
                    if args.llm == 'transformer-encoder-decoder':
                        # print('args.llm:', args.llm)
                        # print('args.output_no_type:', args.output_no_type)
                        if args.output_no_type:
                            netlist, duty_cycle = read_transformer_output_shrink_canonical_output_no_type(inputs, output, args.duty10)
                        else:
                            netlist, duty_cycle = read_transformer_output_shrink_canonical(output, args.duty10, args.typeNidx, args.common_word)
                        label_vout = float(vout[0])
                        label_eff = float(eff[0])
                    else:
                        netlist, duty_cycle = read_LLM_output_shrink_canonical(output)
                        input_strings = inputs.split()
                        label_vout = float(input_strings[11][:len(input_strings[11])-1])
                        label_eff = float(input_strings[13][:len(input_strings[13])-1])
                    
                elif args.baseline_format == 'shrink_canonical_dutycycle':
                    netlist, duty_cycle = read_LLM_output_shrink_canonical_dutycycle(output)
                    input_strings = inputs.split()
                    label_vout = float(input_strings[11][:len(input_strings[11])-1])
                    label_eff = float(input_strings[13][:len(input_strings[13])-1])
                elif args.baseline_format == 'matrix':
                    if args.llm == 'transformer-encoder-decoder':
                        if args.matrix_half:
                            netlist, duty_cycle = read_transformer_matrix_half(inputs, output, args.duty10)
                        else:
                            netlist, duty_cycle = read_transformer_output_mask(inputs, output, args.duty10)
                        label_vout = float(vout[0])
                        label_eff = float(eff[0])
                        # input('transformer-encoder-decoder')
                    else:
                        vertex_id = inputs.find('Vertex')
                        vertex_string = inputs[vertex_id:]
                        duty_id = output.find('Duty cycle:')
                        edge_idx = output.find('Connections')
                        duty_string = output[duty_id:edge_idx-1]
                        edge_string = output[edge_idx:]
                        # print('vertex_string', vertex_string)
                        circuit_str = vertex_string + ' <sep> ' + duty_string + '<sep> ' + edge_string
                        # print('circuit_str', circuit_str)
                        netlist, duty_cycle = read_masked_LLM_output(circuit_str, 'vertex duty edge')
                        input_strings = inputs.split()
                        label_vout = float(input_strings[11][:len(input_strings[11])-1])
                        label_eff = float(input_strings[13][:len(input_strings[13])-1])
                else:
                    raise NotImplementedError
                if label_eff < 0:
                    continue
                # print(netlist, duty_cycle)
                graph = convert_netlist_2_graph(node_tokens, netlist)
                
                brand_new = True
                for i, trn_graph in enumerate(trn_graphs):
                    if duty_cycle == trn_duty_cycles[i] and nx.vf2pp_is_isomorphic(trn_graph, graph, node_label='type'):
                        brand_new = False
                        result = {}
                        result['Vout'] = trn_vouts[i] * 100
                        result['efficiency'] = trn_effs[i]
                        result['result_valid'] = True
                        # print('in train ', result)
                        if trn_effs[i] == -1:
                            result['result_valid'] = False
                        break
                if brand_new:
                    print('This graph is not in the training set')
                    path = os.path.join(args.output_dir, 'sim.cki')
                    result = sim_generation_output(path, inputs, output, args.baseline_format, args.llm, args.duty10, args.typeNidx)
                    # print('sim ', result)
                    datum = {}
                    datum['circuit_str'] = output
                    datum['netlist'] = netlist
                    datum['eff'] = float(result['efficiency'])
                    datum['vout'] = float(result['Vout']) / 100.0
                    if datum['eff'] == -1:
                        datum['result_valid'] = False
                    new_graph_num += 1
                    cir_data.append(datum)             
                    
                d_dict["result"] = result
                if result['result_valid'] == False:
                    invalid_graph_num += 1
                    print('invalid_graph_num')
                    # input()
                    continue
                output_power_ratio = float(result['Vout']) / 100.0
                output_eff = float(result['efficiency'])
                    # eff = eff
                    # vout = vout * (stat_dict['max_vout'] - stat_dict['min_vout']) + stat_dict['min_vout']
                    # eff = eff * (stat_dict['max_eff'] - stat_dict['min_eff']) + stat_dict['min_eff']
                #     output_power_ratio = output_power_ratio * (stat_dict['max_vout'] - stat_dict['min_vout']) + stat_dict['min_vout']
                #     output_eff = output_eff * (stat_dict['max_eff'] - stat_dict['min_eff']) + stat_dict['min_eff']
                # label_vout = float(vout)
                # label_eff = float(eff)
                print('voltage label: ', label_vout, 'output: ', output_power_ratio)
                print('eff     label: ', label_eff, 'output: ', output_eff)
                # input()
                data_generated.append(d_dict)
                scalar_logits.append(output_power_ratio)
                scalar_labels.append(label_vout)
                eff_logits.append(output_eff)
                eff_labels.append(label_eff)
            except:
                print('wrong_graph_num')
                wrong_graph_num += 1
            # input()
            
        if idx % 500 == 0:
            loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
            print('current mse (vout):        ', loss)
            loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
            print('current mse (eff):         ', loss)
            print('current invalid_graph_num: ', invalid_graph_num)
            print('current wrong_graph_num:   ', wrong_graph_num)
            vout_logits_np = np.array(scalar_logits)
            vout_labels_np = np.array(scalar_labels)
            save_logits(vout_logits_np, vout_labels_np, metrics='vout')
            eff_logits_np = np.array(eff_logits)
            eff_labels_np = np.array(eff_labels)
            save_logits(eff_logits_np, eff_labels_np, metrics='eff')
            jdump(data_generated, os.path.join(args.output_dir, 'data_generated.json'))
            cir_d_path = os.path.join(args.text_data_dir, args.LUT_cir_data_name)
            try:
                with open(cir_d_path, 'w') as f:
                    json.dump(cir_data, f)
            except:
                print('Failed to save cir_data')
    print('total loss: ', np.mean(total_loss))
    # plt.hist(total_loss, bins=100)
    # plt.savefig(os.path.join('plot/loss_hist.png'))
    # plt.close()

    input()
    if sim:
        jdump(data_generated, os.path.join(args.output_dir, 'data_generated.json')) 
        print('current invalid_graph_num: ', invalid_graph_num)
        print('current wrong_graph_num:   ', wrong_graph_num)       
        print('current new_graph_num:     ', new_graph_num)
    
    # mse_loss = nn.MSELoss()
    if get_mse or sim:
        loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
        print('current mse (vout):        ', loss)
        loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
        print('current mse (eff):         ', loss)
        print('len of label ', len(scalar_labels))
        scalar_logits = np.array(scalar_logits)
        scalar_labels = np.array(scalar_labels)
        save_logits(scalar_logits, scalar_labels, metrics='vout')
        eff_logits = np.array(eff_logits)
        eff_labels = np.array(eff_labels)
        save_logits(eff_logits, eff_labels, metrics='eff')
    
    # mse_loss = nn.MSELoss()
    # if get_mse or sim:
    #     loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
    #     print('current mse: ', loss)
    #     print('len of label ', len(scalar_labels))
    #     scalar_logits = np.array(scalar_logits)
    #     scalar_labels = np.array(scalar_labels)
    #     with open(os.path.join(args.output_dir, 'scalar_logits.npy'), 'wb') as f:
    #         np.save(f, scalar_logits)
    #     with open(os.path.join(args.output_dir, 'scalar_labels.npy'), 'wb') as f:
    #         np.save(f, scalar_labels)
        
    #     scalar_logits = np.load(os.path.join(args.output_dir, 'scalar_logits.npy'))
    #     scalar_labels = np.load(os.path.join(args.output_dir, 'scalar_labels.npy'))
    #     plt.scatter(scalar_labels, scalar_logits)
    #     plt.xlabel('Power conversion ratio labels')
    #     if get_mse:
    #         plt.ylabel('Power conversion ratio predictions')
    #     else:
    #         plt.ylabel('Power conversion ratio generations')
    #     plt.savefig(args.output_dir + "/logit_label.png", dpi=200)

def finetune_maskedGen(args, model, tokenizer, data_collator, dset_trn, dset_val):
    if args.use_wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    _, args_trn = generate_config_param(args)
    if not args.ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dset_trn,
        eval_dataset=dset_val,
        args=args_trn,
        data_collator=data_collator,
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    # with torch.autocast("cuda"):
    trainer.train()

    model.save_pretrained(args.output_dir)

def combine_masked_input_output(inputs, outputs):
    # input_strings = inputs.split()
    # print('start combine')
    input_strings = []
    output_strings = []
    mask_id = 0
    while True:
        prev_id = '<extra_id_' + str(mask_id) + '>'
        curr_id = '<extra_id_' + str(mask_id+1) + '>'
        st_token_index = outputs.find(prev_id)
        end_token_index = outputs.find(curr_id)
        # print('st_token_index', st_token_index)
        # print('end_token_index', end_token_index)
        if end_token_index == -1:
            end_token_index = len(outputs)
            output_strings.append(outputs[st_token_index+len(prev_id):end_token_index])
            break
        output_strings.append(outputs[st_token_index+len(prev_id):end_token_index])
        mask_id += 1
    # print('output_strings', output_strings)
    num_mask = len(output_strings)
    for i in range(num_mask+1):
        if i == 0:
            st_token_index = 0
            curr_id = '<extra_id_' + str(i) + '>'
            end_token_index = inputs.find(curr_id)
        elif i == num_mask:
            prev_id = '<extra_id_' + str(i-1) + '>'
            st_token_index = inputs.find(prev_id) + len(prev_id)
            end_token_index = len(inputs)
        else:
            prev_id = '<extra_id_' + str(i-1) + '>'
            curr_id = '<extra_id_' + str(i) + '>'
            st_token_index = inputs.find(prev_id) + len(prev_id)
            end_token_index = inputs.find(curr_id)
        input_strings.append(inputs[st_token_index:end_token_index])
    # print('input_strings', input_strings)
    total_strings = []
    for i in range(num_mask+1):
        if i == num_mask:
            total_strings.append(input_strings[i])
        else:
            total_strings.append(input_strings[i])
            total_strings.append(output_strings[i])
    # print('total_strings', total_strings)
    total_string = ' '.join(total_strings)
    total_strings = total_string.split()
    total_string = ' '.join(total_strings)
    return total_string

def denormalize(vout, eff, stat_dict):
    vout = vout * (stat_dict['max_vout'] - stat_dict['min_vout']) + stat_dict['min_vout']
    # eff = eff * (stat_dict['max_eff'] - stat_dict['min_eff']) + stat_dict['min_eff']
    eff = eff
    # vout = vout * torch.max(torch.abs(stat_dict['min_vout']), torch.abs(stat_dict['max_vout']))
    return vout, eff

def report_trn_cir_data(args, cir_data):
    node_tokens = set()
    type_str = ['Sa', 'Sb', 'C', 'L']
    for device in type_str:
        for i in range(20):
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
    for datum in cir_data:
        # try:
        if not args.duty10:
            circuit_str = datum['circuit_str']
            netlist, duty_cycle = read_masked_LLM_output(datum['circuit_str'], args.order)
        else:
            circuit_str = datum['output']
            netlist, duty_cycle = read_transformer_output_mask(datum['input'], datum['output'], args.duty10, pre_eval=True) 
        # except:
        #     print('Error in reading netlist')
        #     continue
        datum['netlist'] = netlist
        graph = convert_netlist_2_graph(node_tokens, netlist)
        trn_graphs.append(graph)
        trn_duty_cycles.append(duty_cycle)
        trn_cir_strs.append(circuit_str)
        trn_effs.append(datum['eff'])
        trn_vouts.append(datum['vout'])
    # trn_effs = np.array(trn_effs)
    # trn_vouts = np.array(trn_vouts)
    return trn_graphs, trn_duty_cycles, trn_cir_strs, trn_effs, trn_vouts, node_tokens

def combine_masked_input_output_encoder(input_ids, output_ids, data_collator):
    duty_cycle_mask_token_id = data_collator.duty_cycle_mask_token_id
    edge_mask_token_id = data_collator.edge_mask_token_id
    node_mask_token_id = data_collator.node_mask_token_id
    masked_set = set([duty_cycle_mask_token_id.item(), edge_mask_token_id.item(), node_mask_token_id.item(), -100])
    # print('masked_set', masked_set)
    # masked_string_set = set(['<duty_cycle_mask>', '<edge_mask>', '<node_mask>'])
    start_id = output_ids.size(1) - input_ids.size(1)
    for i in range(input_ids.size(1)):
        # print(input_ids[0][i].item())
        if input_ids[0][i].item() in masked_set:
            # print('check input_ids[0][i]', input_ids[0][i])
            # print(output_ids[0][start_id+i])
            input_ids[0][i] = output_ids[0][start_id+i].argmax(axis=-1)
    return input_ids
    


def val_maskedGen(args, model, tokenizer, data_collator, dset_trn, dset_val, cir_data, stat_dict=None, get_mse=False, sim=False):
    def save_logits(scalar_logits, scalar_labels, metrics='vout'):
        with open(os.path.join(args.output_dir, 'scalar_logits_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_logits)
        with open(os.path.join(args.output_dir, 'scalar_labels_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_labels)
        
        scalar_logits = np.load(os.path.join(args.output_dir, 'scalar_logits_{}.npy'.format(metrics)))
        scalar_labels = np.load(os.path.join(args.output_dir, 'scalar_labels_{}.npy'.format(metrics)))
        plt.scatter(scalar_labels, scalar_logits)
        plt.xlabel('Voltage conversion ratio labels')
        if get_mse:
            plt.ylabel('Voltage conversion ratio predictions')
        else:
            plt.ylabel('Voltage conversion ratio generations')
        plt.savefig(args.output_dir + "/logit_label_{}.png".format(metrics), dpi=300)
        plt.close()

    trn_graphs, trn_duty_cycles, trn_cir_strs, trn_effs, trn_vouts, node_tokens = report_trn_cir_data(args, cir_data)

    temperature=0.01
    top_p=0.75
    top_k=10
    num_beams=4
    max_new_tokens=256
    stream_output=False
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    generation_param = {
        # "temperature": temperature,
        # 'top_p': top_p,
        # 'top_k': top_k,
        # 'num_beams': num_beams,
        # # 'do_sample': True,
        'max_new_tokens': max_new_tokens,
        # 'repetition_penalty': 1.5,
        'output_scores': True
    }
    wrong_graph_num = 0
    invalid_graph_num = 0
    new_graph_num = 0
    model.eval()
    
    scalar_logits = []
    scalar_labels = []
    eff_logits = []
    eff_labels = []
    data_generated = []
    # expanded_inputs_length, targets_length = compute_input_and_target_lengths(
    #     inputs_length=args.cutoff_len,
    #     noise_density=0.3,
    #     mean_noise_span_length=3,
    # )
    # padding_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, task='conditionalGen')
    # data_collator.data_augment = True
    loader_val = DataLoader(dset_val, batch_size=1,
                shuffle=False, num_workers=0, pin_memory=True, collate_fn=data_collator)
    total_loss = []
    total_batch = 0
    num_data = 0
    for idx, data in enumerate(tqdm(loader_val)):
        total_batch += 1
        # if idx < 13058:
        #     continue
        # if data['n_nodes'][0] != 6:
        #     # print(data['n_nodes'])
        #     continue
        # print(data['n_nodes'])
        input_ids = data["input_ids"].to('cuda')
        # prefixes = data["prefixes"].to('cuda')
        labels = data["labels"].to('cuda')

        vout = data['vout'].to('cuda')
        eff = data['eff'].to('cuda')
        if not args.llm == 'flan-t5-baseline':
            d_cycle_input_ids = data['d_cycle_input_ids'].to('cuda')
            volt_input_ids= data['volt_input_ids'].to('cuda')
            eff_input_ids = data['eff_input_ids'].to('cuda')
            d_cycle_option = data['d_cycle_option'].to('cuda')
        # print('input_ids', input_ids.size())
        # print('labels', labels.size())
        # print('d_cycle_input_ids', d_cycle_input_ids.size())
        # print('volt_input_ids', volt_input_ids.size())
        # print('eff_input_ids', eff_input_ids.size())
        # print('vout', vout.size())
        # print('eff', eff.size())
        # print('d_cycle_option', d_cycle_option.size())

        # if eff.dtype == torch.int64 or vout.dtype == torch.int64:
        #     print('eff', eff)
        #     print('vout', vout)
        #     input()
            # continue
        # if data["n_nodes"][0] != 6:
        #     continue
        # if idx > 10:
        #     break
        with torch.no_grad():
            if args.llm == 'flan-t5':
                # print('use flan-t5')
                # generation_output = model.generate(
                #     input_ids=input_ids,
                #     d_cycle_input_ids=d_cycle_input_ids, 
                #     volt_input_ids=volt_input_ids, eff_input_ids=eff_input_ids, vout=vout, eff=eff, d_cycle_option=d_cycle_option,
                #     generation_config=generation_config,
                #     **generation_param
                # )
                generation_output = model(input_ids=input_ids, labels=labels, d_cycle_input_ids=d_cycle_input_ids, \
                    volt_input_ids=volt_input_ids, eff_input_ids=eff_input_ids, vout=vout, eff=eff, d_cycle_option=d_cycle_option,)
                # generation_output_sample = model(input_ids=input_ids, labels=labels)
                # print(generation_output_sample.loss)
                total_loss.append(generation_output.loss.item())

            elif args.llm == 'flan-t5-baseline':
                # generation_output = model.generate(
                #     input_ids=input_ids,
                #     generation_config=generation_config,
                #     **generation_param
                # )
                generation_output = model(input_ids=input_ids, labels=labels)
                total_loss.append(generation_output.loss.item())
            elif args.llm == 'flan-t5-encoder':
                sim = False
                generation_output = model(input_ids=input_ids, labels=labels, d_cycle_input_ids=d_cycle_input_ids, \
                    volt_input_ids=volt_input_ids, eff_input_ids=eff_input_ids, vout=vout, eff=eff, d_cycle_option=d_cycle_option,)
                # print(generation_output.loss)
                total_loss.append(generation_output.loss.item())
                # generation_output = combine_masked_input_output_encoder(input_ids, generation_output.logits, data_collator)
        bsz = len(input_ids)
        for bs_idx in range(bsz):
            d_dict = {}
            num_data += 1
            inputs = tokenizer.decode(input_ids[bs_idx], skip_special_tokens=False)
            output = tokenizer.decode(generation_output[bs_idx], skip_special_tokens=False)
            label = tokenizer.decode(labels[bs_idx], skip_special_tokens=False)
            d_dict["input"] = inputs
            d_dict["output"] = output
            d_dict["label"] = label
            # print(generation_output)
            # print("Current data id: ", idx)
            # print("Input: ", inputs)
            # print("Output:", output)
            # print("Label: ", label)
            # input()
            # continue
                
            if sim:
                try:
                    if args.mask_style == 'T5':
                        # output = label
                        if args.llm == 'flan-t5-baseline':
                            st_token_index = inputs.find('<extra_id_0>')
                            inputs = inputs[st_token_index-11:]
                        # print('modified inputs', inputs)
                        output = combine_masked_input_output(inputs, output)
                    elif args.mask_style == 'graph_mask':
                        if args.llm == 'flan-t5' or args.llm == 'flan-t5-baseline':
                            output = output[5:] # remove <pad>
                        elif args.llm == 'flan-t5-encoder':
                            output = output
                    else:
                        raise NotImplementedError
                    # print(output)
                    netlist, duty_cycle = read_masked_LLM_output(output, args.order)
                    # print('netlist', netlist, '\n duty_cycle', duty_cycle)
                    graph = convert_netlist_2_graph(node_tokens, netlist)
                    # print('graph', graph)
                    brand_new = True
                    for i, trn_graph in enumerate(trn_graphs):
                        if duty_cycle == trn_duty_cycles[i] and nx.vf2pp_is_isomorphic(trn_graph, graph, node_label='type'):
                            brand_new = False
                            result = {}
                            result['Vout'] = trn_vouts[i] * 100
                            result['efficiency'] = trn_effs[i]
                            result['result_valid'] = True
                            # print('in train ', result)
                            if trn_effs[i] == -1:
                                result['result_valid'] = False
                            break
                    if brand_new:
                        print('This graph is not in the training set')
                        path = os.path.join(args.output_dir, 'sim.cki')
                        result = sim_masked_generation_output(path, output, args.order)
                        # print('sim ', result)
                        datum = {}
                        datum['circuit_str'] = output
                        datum['eff'] = float(result['efficiency'])
                        datum['vout'] = float(result['Vout']) / 100.0
                        if datum['eff'] == -1:
                            datum['result_valid'] = False
                        new_graph_num += 1
                        cir_data.append(datum)             
                        
                    d_dict["result"] = result
                    if result['result_valid'] == False:
                        invalid_graph_num += 1
                        print('invalid_graph_num')
                        # input()
                        continue
                    output_power_ratio = float(result['Vout']) / 100.0
                    output_eff = float(result['efficiency'])
                    if args.normalize:
                        # vout = vout * torch.max(torch.abs(stat_dict['min_vout']), torch.abs(stat_dict['max_vout']))
                        vout, eff = denormalize(vout, eff, stat_dict)
                        # eff = eff
                        # vout = vout * (stat_dict['max_vout'] - stat_dict['min_vout']) + stat_dict['min_vout']
                        # eff = eff * (stat_dict['max_eff'] - stat_dict['min_eff']) + stat_dict['min_eff']
                    #     output_power_ratio = output_power_ratio * (stat_dict['max_vout'] - stat_dict['min_vout']) + stat_dict['min_vout']
                    #     output_eff = output_eff * (stat_dict['max_eff'] - stat_dict['min_eff']) + stat_dict['min_eff']
                    label_vout = float(vout[bs_idx])
                    label_eff = float(eff[bs_idx])
                    print('voltage label: ', label_vout, 'output: ', output_power_ratio)
                    print('eff     label: ', label_eff, 'output: ', output_eff)
                    # input()
                    data_generated.append(d_dict)
                    scalar_logits.append(output_power_ratio)
                    scalar_labels.append(label_vout)
                    eff_logits.append(output_eff)
                    eff_labels.append(label_eff)
                    #TODO extract the label power conversion ratio
                except:
                    print('wrong_graph_num')
                    wrong_graph_num += 1
            # input()
            
        if num_data % 500 == 0:
            print(total_batch, np.mean(total_loss))
            loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
            print('current mse (vout):        ', loss)
            loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
            print('current mse (eff):         ', loss)
            print('current invalid_graph_num: ', invalid_graph_num)
            print('current wrong_graph_num:   ', wrong_graph_num)

    if sim:
        jdump(data_generated, os.path.join(args.output_dir, 'data_generated.json')) 
        print('current invalid_graph_num: ', invalid_graph_num)
        print('current wrong_graph_num:   ', wrong_graph_num)       
        print('current new_graph_num:     ', new_graph_num)
    
    # mse_loss = nn.MSELoss()
    if get_mse or sim:
        loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
        print('current mse (vout):        ', loss)
        loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
        print('current mse (eff):         ', loss)
        print('len of label ', len(scalar_labels))
        scalar_logits = np.array(scalar_logits)
        scalar_labels = np.array(scalar_labels)
        save_logits(scalar_logits, scalar_labels, metrics='vout')
        eff_logits = np.array(eff_logits)
        eff_labels = np.array(eff_labels)
        save_logits(eff_logits, eff_labels, metrics='eff')
        
def val_maskedRegression(args, model, tokenizer, data_collator, dset_trn, dset_val, stat_dict=None, get_mse=False, dset_val_unnorm=None):
    def save_logits(scalar_logits, scalar_labels, metrics='vout'):
        with open(os.path.join(args.output_dir, 'scalar_logits_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_logits)
        with open(os.path.join(args.output_dir, 'scalar_labels_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_labels)
        
        scalar_logits = np.load(os.path.join(args.output_dir, 'scalar_logits_{}.npy'.format(metrics)))
        scalar_labels = np.load(os.path.join(args.output_dir, 'scalar_labels_{}.npy'.format(metrics)))
        plt.scatter(scalar_labels, scalar_logits)
        plt.xlabel('Voltage conversion ratio labels')
        plt.ylabel('Voltage conversion ratio predictions')
        plt.savefig(args.output_dir + "/logit_label_{}.png".format(metrics), dpi=300)
        plt.close()
    
    wrong_graph_num = 0
    invalid_graph_num = 0
    if args.finetune_method == 'lora':
        model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()
    
    vout_logits = []
    vout_labels = []
    eff_logits = []
    eff_labels = []
    loader_val = DataLoader(dset_val, batch_size=16,
                shuffle=False, num_workers=0, pin_memory=True, collate_fn=data_collator)
    for idx, data in enumerate(tqdm(loader_val)):

        input_ids = data["input_ids"].to('cuda')
        # prefixes = data["prefixes"].to('cuda')
        labels = data["labels"].to('cuda')
        d_cycle_input_ids = data['d_cycle_input_ids'].to('cuda')
        volt_input_ids= data['volt_input_ids'].to('cuda')
        eff_input_ids = data['eff_input_ids'].to('cuda')
        vout = data['vout'].to('cuda')
        eff = data['eff'].to('cuda')
        d_cycle_option = data['d_cycle_option'].to('cuda')
        task = data['task']
        if eff.dtype == torch.int64 or vout.dtype == torch.int64:
            continue
        # print('vout: ', vout, 'eff: ', eff)
        # print('labels: ', labels)
        # input()
        # if idx > 10:
        #     break
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, d_cycle_input_ids=d_cycle_input_ids, \
                    volt_input_ids=volt_input_ids, eff_input_ids=eff_input_ids, vout=vout, eff=eff, d_cycle_option=d_cycle_option, task=task)

            for logit, label in zip(outputs.logits, labels):
                # print('logit: ', logit, 'label: ', label)
                # input()
                if args.normalize:
                    logit = denormalize(logit[0], 5, stat_dict)
                    label = denormalize(label[0], 5, stat_dict)
                # print('denorm logit: ', logit, 'label: ', label)
                # label_unnorm = dset_val_unnorm[idx]['labels']
                # print('label_unnorm: ', label_unnorm)
                # print('label: ', label[0])
                # assert(float(label_unnorm[0]) == float(label[0]), 'label mismatch in idx: {} with label_norm {} and label {}'.format(idx, label_unnorm[0], label[0]))

                vout_logits.append(float(logit[0]))
                vout_labels.append(float(label[0]))
                # eff_logits.append(float(logit[1]))
                # eff_labels.append(float(label[1]))
            # print(generation_output.loss)
            # input()
        if idx % 500 == 0:
            loss = nn.MSELoss()(torch.FloatTensor(vout_logits), torch.FloatTensor(vout_labels))
            print('current mse (vout):        ', loss)
            # loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
            # print('current mse (eff):         ', loss)
            vout_logits_np = np.array(vout_logits)
            vout_labels_np = np.array(vout_labels)
            save_logits(vout_logits_np, vout_labels_np, metrics='vout')
            # eff_logits_np = np.array(eff_logits)
            # eff_labels_np = np.array(eff_labels)
            # save_logits(eff_logits_np, eff_labels_np, metrics='eff')
    
    loss = nn.MSELoss()(torch.FloatTensor(vout_logits), torch.FloatTensor(vout_labels))
    print('current mse (vout):        ', loss)
    print('max vout: ', max(vout_labels), 'min vout: ', min(vout_labels))
    # loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
    # print('current mse (eff):         ', loss)
    vout_logits_np = np.array(vout_logits)
    vout_labels_np = np.array(vout_labels)
    save_logits(vout_logits_np, vout_labels_np, metrics='vout')
    # eff_logits_np = np.array(eff_logits)
    # eff_labels_np = np.array(eff_labels)
    # save_logits(eff_logits_np, eff_labels_np, metrics='eff')

def val_maskedGen_custom_input(args, model, tokenizer, stat_dict, cir_data):
    model.eval()
    trn_graphs, trn_duty_cycles, trn_cir_strs, trn_effs, trn_vouts, node_tokens = report_trn_cir_data(args, cir_data)

    temperature=0.01
    top_p=0.75
    top_k=10
    num_beams=4
    max_new_tokens=256
    stream_output=False
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    generation_param = {
        # "temperature": temperature,
        # 'top_p': top_p,
        # 'top_k': top_k,
        # 'num_beams': num_beams,
        # # 'do_sample': True,
        'max_new_tokens': max_new_tokens,
        # 'repetition_penalty': 1.5,
        'output_scores': True
    }
    def save_logits(scalar_logits, scalar_labels, metrics='vout'):
        with open(os.path.join(args.output_dir, 'scalar_logits_custom_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_logits)
        with open(os.path.join(args.output_dir, 'scalar_labels_custom_{}.npy'.format(metrics)), 'wb') as f:
            np.save(f, scalar_labels)
        
        scalar_logits = np.load(os.path.join(args.output_dir, 'scalar_logits_custom_{}.npy'.format(metrics)))
        scalar_labels = np.load(os.path.join(args.output_dir, 'scalar_labels_custom_{}.npy'.format(metrics)))
        plt.scatter(scalar_labels, scalar_logits)
        plt.xlabel('Voltage conversion ratio labels')
        plt.ylabel('Voltage conversion ratio predictions')
        plt.savefig(args.output_dir + "/logit_label_{}.png".format(metrics), dpi=300)
        plt.close()

    label_vouts = np.arange(0, 1.01, 0.01).astype(np.float32)
    label_effs  = np.arange(0.8, 1.0, 0.01).astype(np.float32)
    # print('label_vout: ', label_vouts)
    vout_logits = []
    vout_labels = []
    eff_logits = []
    eff_labels = []
    wrong_graph_num = 0
    invalid_graph_num = 0

    for label_vout in tqdm(label_vouts):
        for label_eff in label_effs:
            input_prompt = "Duty cycle options:"
            d_cycle_input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
            input_prompt = "<sep>Voltage conversion ratio:"
            volt_input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
            input_prompt = "<sep>Efficiency:"
            eff_input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
            label_vout = 0.9839336276054382
            label_eff = 0.9590846300125122
            vout = torch.as_tensor([[label_vout]]).to('cuda')
            eff = torch.as_tensor([[label_eff]]).to('cuda')
            d_cycle_option = torch.as_tensor([[0.1, 0.3, 0.5, 0.7, 0.9]]).to('cuda')
            # 'Duty cycle:<extra_id_0> <sep> Vertex order:<extra_id_1> <sep> Connections:<extra_id_2></s>'
            # inputs = "Duty cycle:<extra_id_0> <sep> Vertex order:<extra_id_1> <sep> Connections:<extra_id_2></s>"
            inputs = "Vertex order: Sb Sb Sb Sa C VIN GND VOUT <sep> Connections:<extra_id_0> Duty cycle:<extra_id_1></s>"
            inputs = "Vertex order: VIN VOUT Sb Sb Sb Sb Sb C GND <sep> Connections:<extra_id_0> Duty cycle:<extra_id_1></s>"
            input_ids = tokenizer(inputs, return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')

            with torch.no_grad():
                generation_output = model.generate(
                    input_ids=input_ids,
                    d_cycle_input_ids=d_cycle_input_ids, 
                    volt_input_ids=volt_input_ids, eff_input_ids=eff_input_ids, vout=vout, eff=eff, d_cycle_option=d_cycle_option,
                    generation_config=generation_config,
                    **generation_param
                )
            d_dict = {}
            inputs = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            output = tokenizer.decode(generation_output[0], skip_special_tokens=False)
            d_dict["input"] = inputs
            d_dict["output"] = output
            print("Input: ", inputs)
            print("Output:", output)
            try:
                output = combine_masked_input_output(inputs, output)
                # print('circuit full description:', output)

                netlist, duty_cycle = read_masked_LLM_output(output, args.order)
                graph = convert_netlist_2_graph(node_tokens, netlist)
                brand_new = True
                for i, trn_graph in enumerate(trn_graphs):
                    if duty_cycle == trn_duty_cycles[i] and nx.vf2pp_is_isomorphic(trn_graph, graph, node_label='type'):
                        brand_new = False
                        result = {}
                        result['Vout'] = trn_vouts[i] * 100
                        result['efficiency'] = trn_effs[i]
                        result['result_valid'] = True
                        if trn_effs[i] == -1:
                            result['result_valid'] = False
                        break
                if brand_new:
                    print('This graph is not in the training set')
                    path = os.path.join(args.output_dir, 'sim_custom.cki')
                    result = sim_masked_generation_output(path, output)
                output_power_ratio = float(result['Vout']) / 100.0
                output_eff = float(result['efficiency'])
                if result['result_valid'] == False:
                    invalid_graph_num += 1
                    continue
                
                if args.normalize:
                    vout, eff = denormalize(vout, eff, stat_dict)
                    label_eff = float(eff)
                    label_vout = float(vout)
                print('voltage label: ', label_vout, 'output: ', output_power_ratio)
                print('eff     label: ', label_eff, 'output: ', output_eff)
                input()
                vout_logits.append(output_power_ratio)
                vout_labels.append(label_vout)
                eff_logits.append(output_eff)
                eff_labels.append(label_eff)
            except:
                print('wrong_graph_num')
                wrong_graph_num += 1
    loss = nn.MSELoss()(torch.FloatTensor(vout_logits), torch.FloatTensor(vout_labels))
    print('current mse (vout):        ', loss)
    print('max vout: ', max(vout_labels), 'min vout: ', min(vout_labels))
    loss = nn.MSELoss()(torch.FloatTensor(eff_logits), torch.FloatTensor(eff_labels))
    print('current mse (eff):         ', loss)
    print('current invalid_graph_num: ', invalid_graph_num)
    print('current wrong_graph_num:   ', wrong_graph_num)
    vout_logits_np = np.array(vout_logits)
    vout_labels_np = np.array(vout_labels)
    save_logits(vout_logits_np, vout_labels_np, metrics='vout')
    eff_logits_np = np.array(eff_logits)
    eff_labels_np = np.array(eff_labels)
    save_logits(eff_logits_np, eff_labels_np, metrics='eff')



def val_custum_input(args, model, tokenizer):
    input_prompt = "### Instruction: Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve the following target power conversion ratio. ### Input: 2 phase-one switches Sa0 and Sa1, 2 phase-two switch Sb0 and Sb1, 1 inductor L0, 1 capacitance C0, a circuit input VIN, a circuit output VOUT, a ground GND. The duty cycle has five options (0.1, 0.3, 0.5, 0.7, 0.9). The target power conversion ratio is 0.59."
    # input_prompt = "### Instruction: Generate a circuit topology and select the duty cycle from the following available circuit components and duty cycle options to achieve the following target power conversion ratio. ### Input: 1 phase-one switch Sa0, 3 phase-two switches Sb1 and Sb2 and Sb0, 1 capacitance C0, a circuit input VIN, a circuit output VOUT, a ground GND. The duty cycle has five options (0.1, 0.3, 0.5, 0.7, 0.9). The target power conversion ratio is 0.59."
    # input_prompt = "### Instruction: Generate a circuit hypergraph representation and select the duty cycle from the following duty cycle options to achieve the following target power conversion ratio. ### Input: The duty cycle has five options (0.1, 0.3, 0.5, 0.7, 0.9). The target power conversion ratio is 0.99."
    input_prompt = "VIN"
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    print(input_ids)
    input()
    temperature=0.01
    top_p=0.75
    top_k=10
    num_beams=4
    max_new_tokens=256
    stream_output=False
    generation_config = GenerationConfig.from_pretrained(args.base_model)
    generation_param = {
        # "temperature": temperature,
        # 'top_p': top_p,
        # 'top_k': top_k,
        # 'num_beams': num_beams,
        # # 'do_sample': True,
        'max_new_tokens': max_new_tokens,
        # 'repetition_penalty': 1.5,
        'output_scores': True
    }
    wrong_graph_num = 0
    invalid_graph_num = 0
    if args.finetune_method == 'lora':
        model = PeftModel.from_pretrained(model, args.output_dir)
    model.eval()
    with torch.no_grad():
    # generation_output = model(input_ids)
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            **generation_param
        )
    output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    print(input_prompt)
    print(output)
    path = os.path.join(args.output_dir, 'sim1.cki')
    # output = "Here's the circuit representation using a hypergraph: Vertices:C1, VIN, GND, Sa0, C0, VOUT, Sb0, Sb1 Hyperedges:(VOUT, Sb1), (VIN, Sa0, C0, C1), (Sa0, Sb1, Sb0), (GND, C1), (C0, Sb0) The duty cycle is set to 0.3."
    result = sim_generation_output(path, output)
    print(result)
    
    
    
def check_data_augment(tokenizer, dset_trn, dset_trn_aug):
    loader_trn = DataLoader(dset_trn, batch_size=1,
                shuffle=False, num_workers=16, pin_memory=True)
    loader_trn_aug = DataLoader(dset_trn_aug, batch_size=1,
                shuffle=False, num_workers=16, pin_memory=True)
    trn_inputs = []
    for idx, data in enumerate(tqdm(loader_trn)):
        input_ids = data["input_ids"].to('cuda')
        labels = data["labels"].to('cuda')
        inputs = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        trn_inputs.append(inputs)
        if idx == 10:
            break
    trn_input_aug = []
    for idx, data in enumerate(tqdm(loader_trn_aug)):
        input_ids = data["input_ids"].to('cuda')
        labels = data["labels"].to('cuda')
        inputs = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        labels = tokenizer.decode(labels[0], skip_special_tokens=True)
        inputs = inputs + "  " + labels
        trn_input_aug.append(inputs)
        if idx == 10:
            break
        
    for i in range(10):
        print('trn')
        print(trn_inputs[i])
        print('trn_aug')
        print(trn_input_aug[i])
        input()
        
    
            
def val_regression(args, model, tokenizer, dset_trn, dset_val):
    temperature=0.1
    top_p=0.75
    top_k=40
    num_beams=4
    max_new_tokens=256
    stream_output=False
    if args.finetune_method == 'lora':
        generation_config = GenerationConfig.from_pretrained(args.base_model)
        generation_param = {
            "temperature": temperature,
            'top_p': top_p,
            'top_k': top_k,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'output_scores': True
        }
        model = PeftModel.from_pretrained(
            model,
            args.output_dir,
        )
    model.eval()
    scalar_logits = []
    scalar_labels = []
    loader_val = DataLoader(dset_trn, batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
    for idx, data in enumerate(tqdm(loader_val)):
        input_ids = data["input_ids"].to('cuda')
        labels = data["labels"].to('cuda')
        # if idx < 2:
        #     continue
        with torch.no_grad():
            logits = model(input_ids).logits
        
        output = tokenizer.decode(input_ids[0])
        print("Current data id: ", idx)
        # print("Input: ", tokenizer.decode(input_ids[0]))
        print("Output:", output)
        print("Label: ", labels[0])
        print("Logit: ", logits[0])
        scalar_logits.append(logits[0])
        scalar_labels.append(labels[0])
        input()
    
    loss = nn.MSELoss()(torch.FloatTensor(scalar_logits), torch.FloatTensor(scalar_labels))
    print('current mse: ', loss)