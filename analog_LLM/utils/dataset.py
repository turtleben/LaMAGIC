import copy
import os
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
from tqdm import tqdm
import torch
import pickle
import transformers
from matplotlib import pyplot as plt
import utils
from torch.utils.data import Dataset
from topo_data_util.topo_analysis.topoGraph import TopoGraph
# from transformers import Trainers


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_input_regression": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input_regression": (
        "{instruction}"
    ),
    "prompt_input_flanT5": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n"
    ),
    "prompt_shrink_input": (
        "{instruction}\n{input}\n"
    ),
}


def smart_tokenizer_and_embedding_resize(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # print('special_tokens_dict', special_tokens_dict)
        
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    # print('len(tokenizer), num_new_tokens: ', len(tokenizer), num_new_tokens)

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_device_token(args,
                    tokenizer: transformers.PreTrainedTokenizer,
                    model: transformers.PreTrainedModel, 
                    transformers_formulation = False
                    ):
    if not transformers_formulation:
        node_tokens = ['VIN', 'VOUT', 'GND', '<no_edge>', '<edge_1>', '<edge_2>', '<both_edges>', '<select>', '<unselect>']
    else:
        if not args.typeNidx:
            node_tokens = ['VIN', 'VOUT', 'GND', '<no_edge>', '<edge_1>', '<edge_2>', '<both_edges>', '<duty_0.1>', '<duty_0.3>', '<duty_0.5>', '<duty_0.7>', '<duty_0.9>']
        else:
            node_tokens = ['VIN', 'VOUT', 'GND', '<duty_0.1>', '<duty_0.2>', '<duty_0.3>', '<duty_0.4>', '<duty_0.5>', '<duty_0.6>', '<duty_0.7>', '<duty_0.8>', '<duty_0.9>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    type_str = ['Sa', 'Sb', 'C', 'L']
    if args.task == 'conditionalGen' and args.baseline_format != 'matrix' and not args.typeNidx:
        for device in type_str:
            for i in range(5):
                device_str = device + str(i)
                node_tokens.append(device_str)
    else:
        node_tokens = node_tokens + type_str
    if args.mask_style == 'graph_mask':
        print('[graph_mask] adding special tokens ...')
        graph_mask_tokens =  ["<duty_cycle_mask>", "<edge_mask>", "<node_mask>"]
        node_tokens = node_tokens + graph_mask_tokens
    print('node_tokens', node_tokens)
    # 
    # print(node_tokens)
    # node_tokens = set(node_tokens) - set(tokenizer.get_vocab().keys())
    # add the tokens to the tokenizer vocabulary
    n_new_tokens = tokenizer.add_tokens(list(node_tokens))

    special_tokens_dict = dict()
    # special_tokens_dict["no_edge_token"] = "<no_edge>"
    if tokenizer.sep_token is None:
        special_tokens_dict["sep_token"] = "<sep>"
    
    n_new_special_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    args.n_new_tokens = n_new_tokens + n_new_special_tokens
    model.resize_token_embeddings(len(tokenizer))
    print('len(tokenizer):', len(tokenizer), 'n_new_special_tokens', n_new_special_tokens, 'n_new_tokens', n_new_tokens)
    print('sep token id:', tokenizer.sep_token_id)
    # print('get_sentinel_token_ids', tokenizer.get_sentinel_token_ids(), len(tokenizer.get_sentinel_token_ids()))


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, config, data: dict, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        list_data_dict = data

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        print('sources:  \n', sources[0])
        print('targets:  \n', targets[0])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        logging.warning("Finish tokenize inputs... ")
        os.makedirs(config.tokenized_data_dir, exist_ok=True)
        d_path = os.path.join(config.tokenized_data_dir, config.tokenized_data)
        with open(d_path, 'wb') as f:
            pickle.dump(data_dict, f)
        with open(d_path, 'rb') as f:
            data_dict = pickle.load(f)
        # print('data_dict["input_ids"][0]: ', data_dict["input_ids"][0])
        output = tokenizer.decode(data_dict["input_ids"][0])
        print('output: ', output)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    
def preprocess_regression(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (sources, sources)]
    input_ids = sources_tokenized["input_ids"]
    return dict(input_ids=input_ids, labels=targets)
    
class RawTextDatasetRegression(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, config, data: dict, tokenizer: transformers.PreTrainedTokenizer):
        super(RawTextDatasetRegression, self).__init__()
        list_data_dict = data

        logging.warning("Formatting inputs...")
        if config.llm == 'llama':
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        elif config.llm == 'bert':
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_regression"], PROMPT_DICT["prompt_no_input_regression"]
        elif config.llm == 'flan-t5':
            prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_flanT5"], PROMPT_DICT["prompt_no_input_regression"]
        else:
            raise NotImplementedError
        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = []
        for example in list_data_dict:
            label_strings = example['output'].split()
            label = float(label_strings[-1][:len(label_strings[-1])-1])
            targets.append(label)
        # targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        targets = torch.from_numpy(np.array(targets)).float()
        print('sources:', sources[0])
        print('targets:', targets[0])

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess_regression(sources, targets, tokenizer)
        
        # dir = '/home/skunk/dataset_power_converter/text_dataset/node2topo/tokenized_alpaca'
        os.makedirs(config.tokenized_data_dir, exist_ok=True)
        d_path = os.path.join(config.tokenized_data_dir, config.tokenized_data)
        with open(d_path, 'wb') as f:
            pickle.dump(data_dict, f)
        with open(d_path, 'rb') as f:
            data_dict = pickle.load(f)
        # print('data_dict["input_ids"][0]: ', data_dict["input_ids"][0])
        output = tokenizer.decode(data_dict["input_ids"][0])
        print('output: ', output)
        print('labels: ', data_dict["labels"][0])

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

def preprocess_flanT5(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples_tokenized = _tokenize_fn(sources, tokenizer)
    input_ids = examples_tokenized["input_ids"]
    examples_tokenized = _tokenize_fn(targets, tokenizer)
    labels = examples_tokenized["input_ids"]
    
    return dict(input_ids=input_ids, labels=labels)   

class RawTextDatasetConditionalGen_Transformer(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, config, data: dict, tokenizer: transformers.PreTrainedTokenizer, save_tokenized_path=None):
        super(RawTextDatasetConditionalGen_Transformer, self).__init__()
        list_data_dict = data
        self.tokenizer = tokenizer
        # sources = [ f"{example['input']} {tokenizer.eos_token}"
        #             for example in list_data_dict]
        # targets = [ f"{example['output']} {tokenizer.eos_token}"
        #             for example in list_data_dict]
        
        total_token_length = 0.0
        self.prefixes = []
        self.n_nodes = []
        self.input_ids = []
        self.label_ids = []
        for d_dict in tqdm(list_data_dict):
            prefix_dict = {}
            prefix_dict['vout'] = torch.as_tensor([d_dict['vout']])
            prefix_dict['eff'] = torch.as_tensor([d_dict['eff']])
            if config.use_duty_cycle_option_prefix:
                try:
                    prefix_dict['d_cycle_option'] = torch.as_tensor(d_dict['d_cycle_option'])
                except:
                    if config.duty10 == 'True':
                        prefix_dict['d_cycle_option'] = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                    else:
                        prefix_dict['d_cycle_option'] = torch.as_tensor([0.1, 0.3, 0.5, 0.7, 0.9])
            # print(d_dict['input'])
            input_ids = self.tokenizer.encode(" ".join(d_dict['input'].split()))
            # input_ids = self.tokenizer(d_dict['input'], padding="longest", return_tensors="pt",
            #             max_length=self.tokenizer.model_max_length, truncation=True,).input_ids[0]
            label_ids = self.tokenizer.encode(" ".join(d_dict['output'].split()))
            if config.tokenizer == "gpt2":
                input_ids = [tid for tid in input_ids if tid != 220]
                label_ids = [tid for tid in label_ids if tid != 220]
            total_token_length += len(label_ids)
            # label_ids = self.tokenizer(d_dict['output'], padding="longest", return_tensors="pt",
            #             max_length=self.tokenizer.model_max_length, truncation=True,).input_ids[0]
            self.input_ids.append(input_ids)
            self.label_ids.append(label_ids)
            self.prefixes.append(prefix_dict)
            inputs = tokenizer.decode(input_ids)
            label = tokenizer.decode(label_ids)
            print('input: ', inputs)
            print('label: ', label)
            print('input_ids', input_ids)
            print('label_ids', label_ids)
            input()
            # print(prefix_dict['vout'])
            # print(prefix_dict['eff'])
            # input('stop')
        assert len(self.input_ids) == len(self.label_ids) == len(self.prefixes)
        print('len(self.input_ids): ', len(self.input_ids))
        print('total_token_length: ', total_token_length / float(len(self.input_ids)))
        # input()
        data_dict = {}
        data_dict['prefixes'] = self.prefixes
        data_dict['n_nodes'] = self.n_nodes
        data_dict['input_ids'] = self.input_ids
        data_dict['labels'] = self.label_ids
        inputs = tokenizer.decode(data_dict["input_ids"][0])
        label = tokenizer.decode(data_dict["labels"][0])
        with open(save_tokenized_path, 'wb') as f:
            pickle.dump(data_dict, f)

        with open(save_tokenized_path, 'rb') as f:
            data_dict = pickle.load(f)
        print('data_dict["input_ids"]: ',len(data_dict["input_ids"]))
        inputs = tokenizer.decode(data_dict["input_ids"][0])
        label = tokenizer.decode(data_dict["labels"][0])
        print('input_ids: ', data_dict["input_ids"][0])
        print('labels: ', data_dict["labels"][0])
        print('input: ', inputs)
        print('label: ', label)
    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        input_ids = self.input_ids[i]
        labels = self.label_ids[i]
        return dict(input_ids=input_ids, labels=labels, prefixes=self.prefixes[i])

class RawTextDatasetConditionalGen(Dataset):
    def __init__(self, config, list_data_dict: dict, tokenizer: transformers.PreTrainedTokenizer, save_tokenized_path=None):
        super(RawTextDatasetConditionalGen, self).__init__()

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input_flanT5"], PROMPT_DICT["prompt_no_input_regression"]
        if config.baseline_format == 'shrink_canonical' or config.baseline_format == 'shrink_canonical_dutycycle' or config.baseline_format == 'shrink_canonical_dutycycle_first' or config.baseline_format == 'matrix':
            print('shrink_canonical')
            prompt_input = PROMPT_DICT["prompt_shrink_input"]    
            # prompt_input = PROMPT_DICT["prompt_no_input_regression"]        
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        # targets = [f"{example['input']} {example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        print('sources:  \n', sources[0])
        print('targets:  \n', targets[0])
        # for example in list_data_dict:
        #     print('example', example)
        #     input()
        input()
        ## help me write the source and target to a file
        with open('source_target.txt', 'w') as f:
            for i in range(101):
                f.write(sources[i])
                f.write(targets[i])
                f.write('\n')        
        # for i in range(100):
        #     print(sources[i])
        #     print(targets[i])
        input('stop')
        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess_flanT5(sources, targets, tokenizer)
        logging.warning("Finish tokenize inputs... ")
        # dir = '/home/skunk/dataset_power_converter/text_dataset/node2topo/tokenized_alpaca'
        # os.makedirs(config.tokenized_data_dir, exist_ok=True)
        # d_path = os.path.join(config.tokenized_data_dir, config.tokenized_data)
        with open(save_tokenized_path, 'wb') as f:
            pickle.dump(data_dict, f)
        with open(save_tokenized_path, 'rb') as f:
            data_dict = pickle.load(f)
        print('data_dict["input_ids"][0]: ', data_dict["input_ids"][0])
        inputs = tokenizer.decode(data_dict["input_ids"][0])
        label = tokenizer.decode(data_dict["labels"][0])
        print('input: ', inputs)
        print('label: ', label)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])
        
# example of data format
'''
Duty cycle options:0.1 0.2 0.3 0.4 0.5 Voltage conversion ratio:[int] Efficiency:[int]<sep>
Duty cycle: <select><unselect><unselect><unselect><unselect><sep>
Vertex order: VIN VOUT GND Sa0 Sa1 Sb0 L0 L1<sep>
Connections: VIN<no_edge><no_edge><no_edge><edge_1><no_edge><no_edge><edge_1><no_edge>
 VOUT<no_edge><no_edge><no_edge><no_edge><no_edge><no_edge><no_edge><edge_1>
 GND<no_edge><no_edge><no_edge><no_edge><no_edge><edge_1><no_edge><no_edge> 
 Sa0<edge_1><no_edge><no_edge><no_edge><edge_2><no_edge><edge_1><edge_2>
 Sa1<no_edge><no_edge><no_edge><edge_1><no_edge><edge_2><edge_2><edge_1>
 Sb0<no_edge><no_edge><edge_1><no_edge><edge_2><no_edge><edge_2><no_edge>
 L0<edge_1><no_edge><no_edge><edge_1><edge_2><edge_2><no_edge><no_edge>
 L1<no_edge><edge_1><no_edge><edge_2><edge_2><no_edge><no_edge><no_edge><sep>
'''
class RawDatasetMaskedGen(Dataset):
    def __init__(self, config, list_data_dict: list, tokenizer: transformers.PreTrainedTokenizer, train=True, save_tokenized_path=None):
        super(RawDatasetMaskedGen, self).__init__()
        self.train = train
        self.masked_method = config.masked_method
        self.masked_ratio = config.masked_ratio
        self.data_dict = list_data_dict
        self.tokenizer = tokenizer

        logging.warning("Formatting and tokenizing inputs...")
        input_prompt = "Duty cycle options:"
        dcycle_input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_prompt = "<sep>Voltage conversion ratio:"
        volt_input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        input_prompt = "<sep>Efficiency:"
        eff_input_ids = tokenizer(input_prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        # print('dcycle_input_ids', dcycle_input_ids)
        # print('volt_input_ids', volt_input_ids)
        # print('eff_input_ids', eff_input_ids)
        sent_token_id = 0
        sent_token = f'<extra_id_{sent_token_id}>'
        self.prefixes = []
        self.n_nodes = []
        self.input_ids = []
        for d_dict in list_data_dict:
            prefix_dict = {}
            nodes = d_dict['list_of_node']
            edges = d_dict['list_of_edge']
            graph = TopoGraph(nodes, edge_list=edges)
            prefix_dict['d_cycle_input_ids'] = dcycle_input_ids
            prefix_dict['volt_input_ids'] = volt_input_ids
            prefix_dict['eff_input_ids'] = eff_input_ids
            prefix_dict['vout'] = torch.as_tensor([d_dict['vout']])
            prefix_dict['eff'] = torch.as_tensor([d_dict['eff']])
            prefix_dict['d_cycle_option'] = torch.as_tensor([0.1, 0.3, 0.5, 0.7, 0.9])
            # input_ids = torch.cat((dcycle_input_ids, torch.as_tensor([0.1, 0.3, 0.5, 0.7, 0.9])), dim=-1)
            # input_ids = torch.cat((input_ids, volt_input_ids, torch.as_tensor([d_dict['vout']])), dim=-1)
            # input_ids = torch.cat((input_ids, eff_input_ids, torch.as_tensor([d_dict['eff']])), dim=-1)
            # print('prefix', input_ids)
            self.prefixes.append(prefix_dict)
            n_node = 0
            for node in nodes:
                if type(node) == int:
                    continue
                n_node += 1
            self.n_nodes.append(n_node)
            input_ids = self.tokenizer(d_dict['circuit_str'], padding="longest", return_tensors="pt",
                        max_length=self.tokenizer.model_max_length, truncation=True,).input_ids[0]
            self.input_ids.append(input_ids)

            # print("Duty cycle:<select>", tokenizer("Duty cycle:<select>", return_tensors="pt").input_ids[0])
            # print("Duty cycle: <select>", tokenizer("Duty cycle: <select>", return_tensors="pt").input_ids[0])
            # print(tokenizer('<no_edge>', return_tensors="pt").input_ids[0])
            # # print(tokenizer('1', return_tensors="pt").input_ids[0])
            # print(tokenizer('<edge_1>', return_tensors="pt").input_ids[0])
            # print(tokenizer('<edge_1><edge_2>', return_tensors="pt").input_ids[0])
            # # encode_seq = tokenizer('Sa0 <no_edge><no_edge><edge_1><no_edge><no_edge><edge_2><edge_2><no_edge><sep>', return_tensors="pt").input_ids[0]
            # encode_seq = tokenizer(d_dict['circuit_str'], return_tensors="pt").input_ids[0]
            # print(encode_seq)
            # input_str = self.tokenizer.decode(encode_seq)
            # print(input_str)

            # print('<sep>', tokenizer('<sep>', return_tensors="pt").input_ids[0])
            # input()
        data_dict = {}
        data_dict['prefixes'] = self.prefixes
        data_dict['n_nodes'] = self.n_nodes
        data_dict['input_ids'] = self.input_ids
        with open(save_tokenized_path, 'wb') as f:
            pickle.dump(data_dict, f)
    
    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        d_dict = self.data_dict[i]
        input_ids = labels = self.input_ids[i]
        # input_ids = labels = self.tokenizer(d_dict['circuit_str'], padding="longest", return_tensors="pt",
        #                 max_length=self.tokenizer.model_max_length, truncation=True,).input_ids[0]

        return dict(input_ids=input_ids, labels=labels, prefixes=self.prefixes[i], n_nodes=self.n_nodes[i])

class NormalizeDatasetMaskGen(Dataset):
    def __init__(self, args, data_dict_trn, data_dict_val):
        data_dict_trn = copy.deepcopy(data_dict_trn)
        data_dict_val = copy.deepcopy(data_dict_val)
        self.prefixes = []
        self.n_nodes = []
        self.input_ids = []
        effs = []
        vouts = []
        for prefix, n_node, input_id in zip(data_dict_trn['prefixes'], data_dict_trn['n_nodes'], data_dict_trn['input_ids']):
            if args.prune_invalid and prefix['eff'][0] < 0:
                continue
            self.prefixes.append(prefix)
            self.n_nodes.append(n_node)
            self.input_ids.append(input_id)
            effs.append(prefix['eff'][0])
            vouts.append(prefix['vout'][0])
        for prefix, n_node, input_id in zip(data_dict_val['prefixes'], data_dict_val['n_nodes'], data_dict_val['input_ids']):
            if args.prune_invalid and prefix['eff'][0] < 0:
                continue
            self.prefixes.append(prefix)
            self.n_nodes.append(n_node)
            self.input_ids.append(input_id)
            effs.append(prefix['eff'][0])
            vouts.append(prefix['vout'][0])
        self.min_eff, self.max_eff = torch.min(torch.as_tensor(effs)), torch.max(torch.as_tensor(effs))
        self.min_vout, self.max_vout = torch.min(torch.as_tensor(vouts)), torch.max(torch.as_tensor(vouts))
        effs = []
        vouts = []
        # print('min_eff', self.min_eff, 'max_eff', self.max_eff)
        # print('min_vout', self.min_vout, 'max_vout', self.max_vout)
        for i in range(len(self.prefixes)):
            # self.prefixes[i]['eff'][0] = (self.prefixes[i]['eff'][0] - self.min_eff) /(self.max_eff-self.min_eff)
            effs.append(self.prefixes[i]['eff'][0])
            self.prefixes[i]['vout'][0] = (self.prefixes[i]['vout'][0] - self.min_vout) / (self.max_vout-self.min_vout)
            # self.prefixes[i]['vout'][0] = self.prefixes[i]['vout'][0] / torch.max(torch.abs(self.min_vout), torch.abs(self.max_vout))
            vouts.append(self.prefixes[i]['vout'][0])
        self.mean_eff = float(torch.mean(torch.as_tensor(effs)))
        self.mean_vout = float(torch.mean(torch.as_tensor(vouts)))
    def return_statistic(self):
        s_dict = {}
        s_dict['min_eff'] = self.min_eff
        s_dict['max_eff'] = self.max_eff
        s_dict['min_vout'] = self.min_vout
        s_dict['max_vout'] = self.max_vout
        s_dict['mean_eff'] = self.mean_eff
        s_dict['mean_vout'] = self.mean_vout
        return s_dict
        

class TokenizedDatasetMaskGen(Dataset):
    def __init__(self, args, data_dict, stat_dict, data_num='all', prune_invalid=True, normalize=True):
        data_dict = copy.deepcopy(data_dict)
        self.prefixes = []
        self.n_nodes = []
        self.input_ids = []
        effs = []
        vouts = []
        dnum = 0
        for prefix, n_node, input_id in zip(data_dict['prefixes'], data_dict['n_nodes'], data_dict['input_ids']):
            dnum += 1
            if args.prune_invalid and prefix['eff'][0] < 0:
                continue
            # if n_node < 8: ########################################################
            #     continue
            #     print('there is node = 5 data')
            #     input()
            self.prefixes.append(prefix)
            self.n_nodes.append(n_node)
            self.input_ids.append(input_id)
            effs.append(prefix['eff'][0])
            vouts.append(prefix['vout'][0])
        print('train data num = ', dnum)
        if args.normalize:
            effs = []
            vouts = []
            # print('min_eff', self.min_eff, 'max_eff', self.max_eff)
            # print('min_vout', self.min_vout, 'max_vout', self.max_vout)
            print('id0 eff = ', self.prefixes[0]['eff'][0])
            print('id0 vout = ', self.prefixes[0]['vout'][0])
            print('id1 eff = ', self.prefixes[1]['eff'][0])
            print('id1 vout = ', self.prefixes[1]['vout'][0])
            for i in range(len(self.prefixes)):
                # self.prefixes[i]['eff'][0] = (self.prefixes[i]['eff'][0] - stat_dict['min_eff']) /(stat_dict['max_eff'] - stat_dict['min_eff'])
                effs.append(self.prefixes[i]['eff'][0])
                # print((self.prefixes[i]['vout'][0] - self.min_vout), (self.max_vout-self.min_vout))
                self.prefixes[i]['vout'][0] = (self.prefixes[i]['vout'][0] - stat_dict['min_vout']) / (stat_dict['max_vout'] - stat_dict['min_vout'])
                # self.prefixes[i]['vout'][0] = self.prefixes[i]['vout'][0] / torch.max(torch.abs(stat_dict['min_vout']), torch.abs(stat_dict['max_vout']))
                vouts.append(self.prefixes[i]['vout'][0])
            print('id0 vout norm = ', self.prefixes[0]['vout'][0])
            # print('id0 vout norm = ', self.prefixes[0]['vout'][0])
        self.mean_eff = stat_dict['mean_eff']
        self.mean_vout = stat_dict['mean_vout']
            # effs = (torch.as_tensor(effs) - self.min_eff) / (self.max_eff-self.min_eff)
            # vouts = (torch.as_tensor(vouts) - self.min_vout) / (self.max_vout-self.min_vout)
            # plt.hist(effs, bins=200)
            # plt.savefig('plot/eff_hist_normalize.png',  dpi=200)
            # plt.close()
            # plt.hist(vouts, bins=200)
            # plt.savefig('plot/vout_hist_normalize.png',  dpi=200)
            # plt.close()
            # input('plot')
            
        if type(data_num) == str and data_num == 'all':
            pass
        elif type(data_num) == float or type(data_num) == int:
            assert(data_num <= len(self.prefixes))
            np.random.seed(13)
            if type(data_num) == int:
                random_num = data_num
            else:
                random_num = int(data_num * len(self.prefixes))
            prefixes, n_nodes, input_ids = [], [], []
            idx = np.random.choice(len(self.prefixes), 500, replace=False)
            print(idx)
            input('stop')
            random_num -= 500
            idx_list = np.arange(len(self.prefixes))
            idx_list = np.setdiff1d(idx_list, idx)
            for i in idx:
                prefixes.append(self.prefixes[i])
                n_nodes.append(self.n_nodes[i])
                input_ids.append(self.input_ids[i])
            while random_num > 0:
                random_num -= 500
                idx = np.random.choice(idx_list, 500, replace=False)
                idx_list = np.setdiff1d(idx_list, idx)
                for i in idx:
                    prefixes.append(self.prefixes[i])
                    n_nodes.append(self.n_nodes[i])
                    input_ids.append(self.input_ids[i])
            self.input_ids = input_ids
            self.prefixes = prefixes
            self.n_nodes = n_nodes
            # print('shrink data to ', len(self.prefixes))
            # input('stop')

            # idx = np.random.choice(len(self.prefixes), random_num, replace=False)
            
            # self.prefixes = [self.prefixes[i] for i in idx]
            # self.n_nodes = [self.n_nodes[i] for i in idx]
            # self.input_ids = [self.input_ids[i] for i in idx]
            # print('shrink data to ', len(self.prefixes))
            # self.prefixes = self.prefixes[:data_num]
            # self.n_nodes = self.n_nodes[:data_num]
            # self.input_ids = self.input_ids[:data_num]
        else:
            raise NotImplementedError
        self.masked_method = args.masked_method
        if self.masked_method == 'regression':
            self.labels = []
            # print('do sanity check')
            for i in range(len(self.prefixes)):
                if args.num_labels == 1:
                    self.labels.append(torch.as_tensor([self.prefixes[i]['vout'][0] + self.prefixes[i]['eff'][0]]))
                elif args.num_labels == 2:
                    self.labels.append(torch.as_tensor([self.prefixes[i]['vout'][0], self.prefixes[i]['eff'][0]]))
                else:
                    raise NotImplementedError
                # self.labels.append(torch.as_tensor([self.prefixes[i]['vout'][0]]))
                self.prefixes[i]['vout'] = torch.as_tensor([self.mean_vout])
                self.prefixes[i]['eff'] = torch.as_tensor([self.mean_eff])
            # print('labels in TokenizedDatasetMaskGen', self.labels[0])

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.masked_method == 'regression':
            labels = self.labels[i]
            # assert(labels[0] == self.prefixes[i]['vout'][0])
            input_ids = self.input_ids[i]
        else:
            input_ids = labels = self.input_ids[i]
        return dict(input_ids=input_ids, labels=labels, prefixes=self.prefixes[i], n_nodes=self.n_nodes[i])
    
class TokenizedDataset_Transformer(Dataset):
    def __init__(self, args, data_dict, stat_dict, data_num='all', prune_invalid=True, normalize=False):
        data_dict = copy.deepcopy(data_dict)
        self.input_ids = []
        self.labels = []
        self.prefixes = []
        effs = []
        vouts = []
        dnum = 0
        print('len of data_dict', len(data_dict['input_ids']))
        for prefix, input_ids, labels in zip(data_dict['prefixes'], data_dict['input_ids'], data_dict['labels']):
            dnum += 1
            # print(prefix['eff'][0])
            if args.prune_invalid and prefix['eff'][0] < 0:
                continue
            if args.use_duty_cycle_option_prefix == True and args.typeNidx == True:
                if args.duty10 == True:
                    prefix['d_cycle_option'] = torch.as_tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                else:
                    prefix['d_cycle_option'] = torch.as_tensor([0.1, 0.3, 0.5, 0.7, 0.9])
            self.prefixes.append(prefix)
            self.input_ids.append(input_ids)
            self.labels.append(labels)
            effs.append(prefix['eff'][0])
            vouts.append(prefix['vout'][0])
        assert len(self.input_ids) == len(self.labels) == len(self.prefixes)
        print('total train data num = ', dnum)
        

        if type(data_num) == str and data_num == 'all':
            pass
        elif type(data_num) == float or type(data_num) == int:
            assert(data_num <= len(self.prefixes))
            np.random.seed(13)
            if type(data_num) == int:
                random_num = data_num
            else:
                random_num = int(data_num * len(self.prefixes))
            prefixes, labels, input_ids = [], [], []
            idx = np.random.choice(len(self.prefixes), 500, replace=False)
            random_num -= 500
            idx_list = np.arange(len(self.prefixes))
            idx_list = np.setdiff1d(idx_list, idx)
            for i in idx:
                prefixes.append(self.prefixes[i])
                labels.append(self.labels[i])
                input_ids.append(self.input_ids[i])
            while random_num > 0:
                random_num -= 500
                idx = np.random.choice(idx_list, 500, replace=False)
                idx_list = np.setdiff1d(idx_list, idx)
                for i in idx:
                    prefixes.append(self.prefixes[i])
                    labels.append(self.labels[i])
                    input_ids.append(self.input_ids[i])
            self.input_ids = input_ids
            self.prefixes = prefixes
            self.labels = labels
            print('current training data number', len(self.prefixes))
        else:
            raise NotImplementedError
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=torch.tensor(self.input_ids[i]), labels=torch.tensor(self.labels[i]), prefixes=self.prefixes[i])

class TokenizedDataset(Dataset):
    def __init__(self, args, inputs, labels, data_num='all', tokenizer=None):
        self.input_ids = []
        self.labels = []
        # if tokenizer is not None:
        #     for input_ids, labels in tqdm(zip(inputs, labels)):
        #         input_string = tokenizer.decode(input_ids, skip_special_tokens=True)
        #         # print(input_string)
        #         input_strings = input_string.split()
        #         if args.baseline_format == "original":
        #             label_eff = float(input_strings[-1][:len(input_strings[-1])-1])
        #         else:
        #             # label_vout = float(input_strings[11][:len(input_strings[11])-1])
        #             label_eff = float(input_strings[13][:len(input_strings[13])-1])
        #         # print(label_eff)
        #         if label_eff < 0:
        #             # input('stop')
        #             continue
        #         self.input_ids.append(input_ids)
        #         self.labels.append(labels)
        # # print('len(self.input_ids)', len(self.input_ids))
        # # print('len(self.labels)', len(self.labels))
        # # input('stop')
        # if type(data_num) == str and data_num == 'all':
        #     pass
        # elif type(data_num) == int:
        #     assert(data_num <= len(inputs))
        #     np.random.seed(13)
        #     if type(data_num) == int:
        #         random_num = data_num
        #     else:
        #         random_num = int(data_num * len(self.input_ids))
        #     labels, input_ids = [], []
        #     idx = np.random.choice(len(self.input_ids), 500, replace=False)
        #     # print(idx)
        #     # input('stop')
        #     random_num -= 500
        #     idx_list = np.arange(len(self.input_ids))
        #     idx_list = np.setdiff1d(idx_list, idx)
        #     for i in idx:
        #         labels.append(self.labels[i])
        #         input_ids.append(self.input_ids[i])
        #     while random_num > 0:
        #         random_num -= 500
        #         idx = np.random.choice(idx_list, 500, replace=False)
        #         idx_list = np.setdiff1d(idx_list, idx)
        #         for i in idx:
        #             labels.append(self.labels[i])
        #             input_ids.append(self.input_ids[i])
        #     self.input_ids = input_ids
        #     self.labels = labels
        #     print('len(self.input_ids)', len(self.input_ids))
        #     print('len(self.labels)', len(self.labels))

        self.input_ids = inputs
        self.labels = labels
        if type(data_num) == str and data_num == 'all':
            self.input_ids = inputs
            self.labels = labels
        elif type(data_num) == int:
            assert(data_num <= len(inputs))
            np.random.seed(13)
            if type(data_num) == int:
                random_num = data_num
            else:
                random_num = int(data_num * len(inputs))
            idx = np.random.choice(len(inputs), random_num, replace=False)
            self.input_ids = [self.input_ids[i] for i in idx]
            self.labels = [self.labels[i] for i in idx]
            # self.input_ids = inputs[:data_num]
            # self.labels = labels[:data_num]
        else:
            raise NotImplementedError

        # self.labels = torch.from_numpy(np.array(labels[:10000])).float()
        # for data in self.labels:
        #     if data >= 3 or data < 0:
        #         print(data)
        
        if args.generate == True and args.task == 'causal':
            for i in range(len(self.input_ids)):
                labels = np.array(self.labels[i])
                input_lens = np.count_nonzero(labels==IGNORE_INDEX)
                self.input_ids[i] = self.input_ids[i].numpy()[:input_lens]
                self.labels[i] = self.labels[i].numpy()[input_lens:]
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    task: str

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # print(instances)
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        if self.task == 'causal' or self.task == 'conditionalGen' or self.task == 'MaskedGen':
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        else:
            labels = torch.stack(labels)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
    
    

                #             if node_number_count == 0:
                #                 input_string_list.append(f'<extra_id_{input_sent_token_id}>')
                #                 input_sent_token_id += 1
                #             node_number_count += 1
                #             label_string_list.append(string) # TODO Need to deal with , and . 
                #             if node_number_count > node_number:
                #                 node_number_count = 0
                #                 edgeID += 1

                    
                # if string == "Connections:":
                #     start_finish_prefix = True
                #     label_string_list.append(f'<extra_id_{label_sent_token_id}>')
                #     label_sent_token_id += 1
                # if not start_finish_prefix:
                #     input_string_list.append(string)
                #     continue
                # if string == "Duty":
                #     assert(edgeID == len(masked_or_not)-1)
                #     node_number = 6 # The number of duty cycle options
                #     if masked_or_not[edgeID] == 1:
                #         label_string_list.append(string)
                #     else:
                #         input_string_list.append(string)
                #     continue
                # if masked_or_not[edgeID] == 1:
                #     if node_number_count == 0:
                #         input_string_list.append(f'<extra_id_{input_sent_token_id}>')
                #         input_sent_token_id += 1
                #     node_number_count += 1
                #     label_string_list.append(string) # TODO Need to deal with , and . 
                #     if node_number_count > node_number:
                #         node_number_count = 0
                #         edgeID += 1
                # else:
                #     if node_number_count == 0: # The first time to add masked token
                #         label_string_list.append(f'<extra_id_{label_sent_token_id}>')
                #         label_sent_token_id += 1
                #     node_number_count += 1
                #     input_string_list.append(string) 
                #     if node_number_count > node_number:
                #         node_number_count = 0
                #         edgeID += 1