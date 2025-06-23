import json
import logging
import math
import os
import sys
import time
import warnings
import copy
import re
from dataclasses import asdict, dataclass, field

# You can also adapt this script on your own masked language modeling task. Pointers for this are left as comments.
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
from parsers.simulation import read_transformer_output_mask, sim_netlist_duty_cycle
import torch

from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_MASKED_LM_MAPPING,
    AutoTokenizer,
    BatchEncoding,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    T5Config,
    is_tensorboard_available,
    set_seed,
    T5Tokenizer,
)
# from transformers.models.t5.modeling_flax_t5 import shift_tokens_right

IGNORE_INDEX = -100

def shift_tokens_right(input_ids: np.array, pad_token_id: int, decoder_start_token_id: int) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(shifted_input_ids == -100, pad_token_id, shifted_input_ids)
    return shifted_input_ids

def compute_input_and_target_lengths(inputs_length, noise_density, mean_noise_span_length):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .

    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.

    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0] <= inputs_length:
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(tokens_length)

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length

@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: PreTrainedTokenizerBase
    masked_method: str
    data_order: str
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int
    n_new_tokens: int
    data_augment: bool = False
    llm: str = 'flan-T5'

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        # batch = BatchEncoding(
        #     {k: np.array([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        # )
        input_ids, labels, prefixes, n_nodes = tuple([instance[key] for instance in examples] for key in ("input_ids", "labels", "prefixes", "n_nodes"))
        # input_str = self.tokenizer.batch_decode(labels)
        # # print('####### input_str', input_str)
        # batch = examples
        # input_ids = batch["input_ids"]
        batch = dict()
        batch['d_cycle_input_ids'] = torch.stack([prefixes[i]['d_cycle_input_ids'] for i in range(len(prefixes))])
        batch['volt_input_ids'] = torch.stack([prefixes[i]['volt_input_ids'] for i in range(len(prefixes))])
        batch['eff_input_ids'] = torch.stack([prefixes[i]['eff_input_ids'] for i in range(len(prefixes))])
        batch['vout'] = torch.stack([prefixes[i]['vout'] for i in range(len(prefixes))])
        batch['eff'] = torch.stack([prefixes[i]['eff'] for i in range(len(prefixes))])
        batch['d_cycle_option'] = torch.stack([prefixes[i]['d_cycle_option'] for i in range(len(prefixes))])
        # batch['n_nodes'] = torch.stack([n_nodes[i] for i in range(len(n_nodes))])
        batch['n_nodes'] = n_nodes


        if self.data_augment:
            np.random.seed(None)
            # print('before data augment')
            # # print(input_ids)
            # input_str = self.tokenizer.batch_decode(input_ids)
            # label_str = self.tokenizer.batch_decode(labels)
            # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
            input_ids_list = []
            labels_list = []
            for i in range(len(input_ids)):
                input_ids_i = input_ids[i]
                random_order = np.random.permutation(n_nodes[i])
                if self.data_order == 'duty vertex edge':
                    graph_start_id = 13
                elif self.data_order == 'vertex edge duty':
                    graph_start_id = 4
                else:
                    raise ValueError('data_order should be duty vertex edge or vertex edge duty')
                    # duty_ids = input_ids_i[0:8]
                node_ids = input_ids_i[graph_start_id:graph_start_id+n_nodes[i]]
                start_id = graph_start_id + n_nodes[i] + 3 + 1
                node_ids = []
                edge_ids = []
                for j in range(n_nodes[i]):
                    node_ids.append(input_ids_i[start_id + j*(n_nodes[i] + 1)])
                    start_mask_id = start_id + j*(n_nodes[i]+1) + 1
                    edge_ids.append(input_ids_i[start_mask_id:start_mask_id+n_nodes[i]][random_order])
                node_ids = np.array(node_ids)[random_order]
                # print('node_ids', node_ids)
                edge_ids = np.array(edge_ids)[random_order]
                # print('edge_ids', edge_ids)
                input_ids_i[graph_start_id:graph_start_id+n_nodes[i]] = torch.from_numpy(node_ids)
                for j in range(n_nodes[i]):
                    input_ids_i[start_id + j*(n_nodes[i]+1)] = node_ids[j]
                    start_mask_id = start_id + j*(n_nodes[i]+1) + 1
                    input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = torch.from_numpy(edge_ids[j])
                labels_i = input_ids_i
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            if not self.masked_method == 'regression':
                labels = torch.nn.utils.rnn.pad_sequence(
                    labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            
            # input_str = self.tokenizer.batch_decode(input_ids)
            # label_str = self.tokenizer.batch_decode(labels)
            # print('after_data_augment')
            # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
            # input()

        if self.masked_method == 'mix':
            p = np.random.rand()
            if p < 0.5:
                cur_masked_method = 'random'
            else:
                cur_masked_method = 'edge-wise'
        elif self.masked_method == 'all-mix':
            p = np.random.rand()
            if p < 0.2:
                cur_masked_method = 'consecutive-random'
            elif p < 0.4:
                cur_masked_method = 'edge-wise'
            elif p < 0.6:
                cur_masked_method = 'full-connection'
            elif p < 0.8:
                cur_masked_method = 'full-graph'
            else:
                cur_masked_method = 'full-graph-no-duty'
        elif self.masked_method == 'mix-graph':
            p = np.random.rand()
            if p < 0.33:
                cur_masked_method = 'full-connection'
            elif p < 0.66:
                cur_masked_method = 'full-graph'
            else:
                cur_masked_method = 'full-graph-no-duty'
        else:
            cur_masked_method = self.masked_method
        print_or_not = False
        batch_size = len(input_ids)
        if cur_masked_method == 'random':
            input_ids = labels = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch_size, expandend_input_length = input_ids.shape

            mask_indices = np.asarray([self.random_spans_noise_mask(expandend_input_length) for i in range(batch_size)])
            labels_mask = ~mask_indices

            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            # # print('####### input_ids_sentinel', input_ids_sentinel)
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            input_ids = torch.tensor(self.filter_input_ids(input_ids, input_ids_sentinel))
            labels = torch.tensor(self.filter_input_ids(labels, labels_sentinel))
        
        elif cur_masked_method == 'consecutive-random':
            # print_or_not = True
            input_ids = labels = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            batch_size, expandend_input_length = input_ids.shape
            mask_indices = np.zeros((batch_size, expandend_input_length,))
            num_mask = np.random.randint(1, expandend_input_length+1)
            mask_indices[:, expandend_input_length-num_mask:] = 1
            mask_indices = mask_indices != 0
            # print('####### node_mask', mask_indices.shape)
            # print(mask_indices)
            labels_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            # # print('####### input_ids_sentinel', input_ids_sentinel)
            labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))
            # print('####### labels_sentinel', labels_mask, labels_sentinel)
            input_ids = torch.tensor(self.filter_input_ids(input_ids, input_ids_sentinel))
            labels = torch.tensor(self.filter_input_ids(labels, labels_sentinel))


        elif cur_masked_method == 'edge-wise':
            mask_indices = []
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                # if n_nodes[i] == 5:
                #     print_or_not = True
                input_ids_i = labels_i = input_ids[i].reshape(1, -1)
                node_mask = np.zeros((n_nodes[i]+1,))
                node_mask[:int( (n_nodes[i]+1) * self.noise_density)] = 1
                np.random.shuffle(node_mask)
                # print(int( (n_nodes[i]+1) * self.noise_density))
                # print(self.noise_density)
                # print('node_mask', node_mask)
                # node_mask = np.random.choice([0, 1], size=(n_nodes[i]+1,), p=[1-self.noise_density, self.noise_density])
                # node_mask = np.asarray([1, 0, 0, 1, 0, 1, 0, 0, 1])
                # if i == 0:
                #     node_mask = np.asarray([1, 1, 0, 1, 1, 1, 0, 1, 1])
                # elif i == 1:
                #     node_mask = np.asarray([0, 1, 1, 1, 0, 1, 0, 0, 1])
                # # print('node_mask', node_mask)
                input_ids_mask = np.zeros((input_ids_i.shape[1],), dtype=np.int8)
                start_id = 13 + n_nodes[i] + 3 + 1 + 1
                for j in range(n_nodes[i]+1):
                    if j == 0:
                        if node_mask[j] == 1:
                            input_ids_mask[3:8] = 1
                    else:
                        if node_mask[j] == 1:
                            start_mask_id = start_id + (j-1)*(n_nodes[i]+1)
                            input_ids_mask[start_mask_id:start_mask_id+n_nodes[i]] = 1
                mask_indices = []
                mask_indices.append(input_ids_mask != 0)
                mask_indices = np.asarray(mask_indices)
                # # print('input_ids_i.shape', input_ids_i.shape)
                # # print('mask_indices', mask_indices)
                labels_mask = ~mask_indices
                input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
                # # print('input_ids_sentinel', input_ids_sentinel)
                # # print('####### input_ids_sentinel', input_ids_sentinel.shape)
                labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

                input_ids_i = torch.tensor(self.filter_input_ids(input_ids_i, input_ids_sentinel))
                labels_i = torch.tensor(self.filter_input_ids(labels_i, labels_sentinel))
                # # print('input_ids_i.shape', input_ids_i.shape)
                # input_str = self.tokenizer.batch_decode(input_ids_i)
                # # print('####### input_ids_i_str', input_str)

                input_ids_list.append(input_ids_i[0])
                labels_list.append(labels_i[0])
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        elif cur_masked_method == "full-connection":
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = labels_i = input_ids[i].reshape(1, -1)
                input_ids_mask = np.zeros((input_ids_i.shape[1],), dtype=np.int8)
                if self.data_order == 'duty vertex edge':
                    start_id = 13 + n_nodes[i] + 3 + 1
                    input_ids_mask[3:8] = 1
                    input_ids_mask[start_id: input_ids_i.shape[1]] = 1
                elif self.data_order == 'vertex edge duty':
                    start_id = 4 + n_nodes[i] + 1 + 3
                    input_ids_mask[start_id: input_ids_i.shape[1]] = 1
                    # new add graph connection
                    input_ids_mask[input_ids_i.shape[1] - 10: input_ids_i.shape[1] - 7] = 0
                mask_indices = []
                mask_indices.append(input_ids_mask != 0)
                mask_indices = np.asarray(mask_indices)
                labels_mask = ~mask_indices
                input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
                labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

                input_ids_i = torch.tensor(self.filter_input_ids(input_ids_i, input_ids_sentinel))
                labels_i = torch.tensor(self.filter_input_ids(labels_i, labels_sentinel))

                input_ids_list.append(input_ids_i[0])
                labels_list.append(labels_i[0])
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        elif cur_masked_method == "full-graph":
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = labels_i = input_ids[i].reshape(1, -1)
                input_ids_mask = np.zeros((input_ids_i.shape[1],), dtype=np.int8)
                if self.data_order == 'duty vertex edge':
                    input_ids_mask[13:13+n_nodes[i]] = 1
                    input_ids_mask[3:8] = 1
                    start_id = 13 + n_nodes[i] + 3 + 1
                    input_ids_mask[start_id: input_ids_i.shape[1]] = 1
                elif self.data_order == 'vertex edge duty':
                    input_ids_mask[4:4+n_nodes[i]] = 1
                    start_id = 4 + n_nodes[i] + 1 + 3
                    input_ids_mask[start_id: input_ids_i.shape[1]] = 1
                    input_ids_mask[input_ids_i.shape[1] - 10: input_ids_i.shape[1] - 7] = 0
                # input_ids_mask = np.ones((input_ids_i.shape[1],), dtype=np.int8)
                mask_indices = []
                mask_indices.append(input_ids_mask != 0)
                mask_indices = np.asarray(mask_indices)
                labels_mask = ~mask_indices
                input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
                labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

                input_ids_i = torch.tensor(self.filter_input_ids(input_ids_i, input_ids_sentinel))
                labels_i = torch.tensor(self.filter_input_ids(labels_i, labels_sentinel))

                input_ids_list.append(input_ids_i[0])
                labels_list.append(labels_i[0])
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        elif cur_masked_method == "full-graph-no-duty":
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = labels_i = input_ids[i].reshape(1, -1)
                input_ids_mask = np.zeros((input_ids_i.shape[1],), dtype=np.int8)
                if self.data_order == 'duty vertex edge':
                    input_ids_mask[13:13+n_nodes[i]] = 1
                    start_id = 13 + n_nodes[i] + 3 + 1
                    input_ids_mask[start_id: input_ids_i.shape[1]] = 1
                elif self.data_order == 'vertex edge duty':
                    start_id = 0
                    input_ids_mask[start_id: input_ids_i.shape[1] - 8] = 1
                mask_indices = []
                mask_indices.append(input_ids_mask != 0)
                mask_indices = np.asarray(mask_indices)
                labels_mask = ~mask_indices
                input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
                labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

                input_ids_i = torch.tensor(self.filter_input_ids(input_ids_i, input_ids_sentinel))
                labels_i = torch.tensor(self.filter_input_ids(labels_i, labels_sentinel))

                input_ids_list.append(input_ids_i[0])
                labels_list.append(labels_i[0])
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
        elif cur_masked_method == "regression":
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = torch.stack(labels)
            # print('input_ids', input_ids)
            # print('labels', labels)
            # input("pause")
            batch["task"] = 'regression'
        else:
            raise ValueError("Invalid masked method")
        # if batch["input_ids"].shape[-1] != self.input_length:
        #     raise ValueError(
        #         f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
        #         f" should be {self.input_length}."
        #     )

        # if batch["labels"].shape[-1] != self.target_length:
        #     raise ValueError(
        #         f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
        #         f" {self.target_length}."
        #     )
        # if print_or_not:
        #     if n_nodes[0] == 5:
        # input_str = self.tokenizer.batch_decode(input_ids)
        # label_str = self.tokenizer.batch_decode(labels)
        # print('input_ids', input_ids)
        # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
        # input()
        if self.llm == 'flan-t5-baseline':

            vout_list = [str(round(prefixes[i]['vout'][0].item(), 6)) for i in range(len(prefixes))]
            eff_list = [str(round(prefixes[i]['eff'][0].item(), 6)) for i in range(len(prefixes))]
            dcycle_float_string = "0.1, 0.3, 0.5, 0.7, 0.9"
            batch['d_cycle_option'] = [ dcycle_float_string for i in range(len(prefixes))]
            # print(batch['d_cycle_option'], batch['vout'], batch['eff'])

            dcycle_float_embeds = self.tokenizer(text=batch['d_cycle_option'], padding=False, return_tensors="pt",
                            add_special_tokens=False, truncation=True).input_ids
            input_ids_list = []
            for i in range(len(dcycle_float_embeds)):
                vout_str = vout_list[i]
                vout_float_embed = self.tokenizer(text=vout_str, padding=False, return_tensors="pt",
                                add_special_tokens=False, truncation=True).input_ids[0]
                eff_str = eff_list[i]
                eff_float_embed = self.tokenizer(text=eff_str, padding=False, return_tensors="pt",add_special_tokens=False, truncation=True).input_ids[0]

                input_ids_i = torch.cat([batch['d_cycle_input_ids'][i], dcycle_float_embeds[i], batch['volt_input_ids'][i], \
                                        vout_float_embed, batch['eff_input_ids'][i], eff_float_embed,  input_ids[i]], dim=-1)
                input_ids_list.append(input_ids_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
            # batch = dict()
        batch["labels"] = labels
        batch["input_ids"] = input_ids
        # input_str = self.tokenizer.batch_decode(input_ids)
        # label_str = self.tokenizer.batch_decode(labels)
        # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
        # input()

        # batch["prefixes"] = torch.stack(prefixes)
        # # print('prefixes', '\n', batch["prefixes"].size())
        # # print('input_ids', '\n', batch["input_ids"])
        # batch["decoder_input_ids"] = torch.tensor(
        #     shift_tokens_right(
        #         batch["labels"], self.pad_token_id, self.decoder_start_token_id
        #     )
        # )
        # input_str = self.tokenizer.batch_decode(batch["decoder_input_ids"])
        # # print('decoder_input_ids', '\n',  input_str)
        # for key, value in batch.items():
        #     batch[key] = torch.tensor(value)

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (len(self.tokenizer) - self.n_new_tokens - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # print(input_ids_full.shape)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        # # print('input_ids_full[input_ids_full >= 0]', input_ids_full[input_ids_full >= 0].shape)
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [input_ids, np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32)], axis=-1
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        num_nonnoise_tokens = length - num_noise_tokens
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        # num_noise_tokens should be less than num_noise_tokens and num_nonnoise_tokens
        num_noise_spans = int(np.round(min(num_noise_tokens, num_nonnoise_tokens) / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            # # print('mask_indices', mask_indices)
            np.random.shuffle(mask_indices)
            # # print('mask_indices', mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            # print('first_in_segment', first_in_segment)
            segment_id = np.cumsum(first_in_segment)
            # print('segment_id', segment_id)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        # print('noise_span_lengths', noise_span_lengths)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans)
        # print('nonnoise_span_lengths', nonnoise_span_lengths)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
        )
        # print('interleaved_span_lengths', interleaved_span_lengths)
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        # print('span_starts', span_starts)
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        # print('span_start_indicator', span_start_indicator)
        span_num = np.cumsum(span_start_indicator)
        # print('span_num', span_num)
        is_noise = np.equal(span_num % 2, 1)
        # print('is_noise', is_noise)

        return is_noise[:orig_length]


class DataCollatorForGraphMLM():

    def __init__(self, tokenizer, masked_method, data_order, noise_density, mean_noise_span_length, input_length, \
                target_length, pad_token_id, decoder_start_token_id, n_new_tokens, llm, data_augment=False):
        print("[graph_mask], init DataCollatorForGraphMLM ...")
        self.tokenizer = tokenizer
        self.masked_method = masked_method
        self.data_order = data_order
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.n_new_tokens = n_new_tokens
        self.data_augment = data_augment
        self.llm = llm
        # self.model = model

        self.duty_cycle_mask_token_id = tokenizer("<duty_cycle_mask>", return_tensors="pt", add_special_tokens=False).input_ids[0][0]
        self.edge_mask_token_id = tokenizer("<edge_mask>", return_tensors="pt", add_special_tokens=False).input_ids[0][0]
        self.node_mask_token_id = tokenizer("<node_mask>", return_tensors="pt", add_special_tokens=False).input_ids[0][0]
        self.sentinel_ids = []
        for i in range(0, 100):
            string_sentinel = "<extra_id_" + str(i) + ">"
            self.sentinel_ids.append(tokenizer(string_sentinel, return_tensors="pt", add_special_tokens=False).input_ids[0][0].item())
        print(self.sentinel_ids)
        # print(self.sentinel_ids, tokenizer("<extra_id_0>", return_tensors="pt", add_special_tokens=False).input_ids[0][0])
        print('Duty_cycle_mask_id: ', self.duty_cycle_mask_token_id, ', Edge_mask_token_id: ', self.edge_mask_token_id, ', Node_mask_token_id: ', self.node_mask_token_id)
        # input('wait')

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        input_ids, labels, prefixes, n_nodes = tuple([instance[key] for instance in examples] for key in ("input_ids", "labels", "prefixes", "n_nodes"))

        batch = dict()
        batch['d_cycle_input_ids'] = torch.stack([prefixes[i]['d_cycle_input_ids'] for i in range(len(prefixes))])
        batch['volt_input_ids'] = torch.stack([prefixes[i]['volt_input_ids'] for i in range(len(prefixes))])
        batch['eff_input_ids'] = torch.stack([prefixes[i]['eff_input_ids'] for i in range(len(prefixes))])
        batch['vout'] = torch.stack([prefixes[i]['vout'] for i in range(len(prefixes))])
        batch['eff'] = torch.stack([prefixes[i]['eff'] for i in range(len(prefixes))])
        batch['d_cycle_option'] = torch.stack([prefixes[i]['d_cycle_option'] for i in range(len(prefixes))])
        # batch['n_nodes'] = n_nodes


        if self.data_augment:
            np.random.seed(None)
            # print('before data augment')
            # print(input_ids)
            # input_str = self.tokenizer.batch_decode(input_ids)
            # label_str = self.tokenizer.batch_decode(labels)
            # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
            input_ids_list = []
            labels_list = []
            for i in range(len(input_ids)):
                input_ids_i = input_ids[i]
                random_order = np.random.permutation(n_nodes[i])
                if self.data_order == 'duty vertex edge':
                    graph_start_id = 13
                elif self.data_order == 'vertex edge duty':
                    graph_start_id = 4
                else:
                    raise ValueError('data_order should be duty vertex edge or vertex edge duty')
                    # duty_ids = input_ids_i[0:8]
                node_ids = input_ids_i[graph_start_id:graph_start_id+n_nodes[i]]
                start_id = graph_start_id + n_nodes[i] + 3 + 1 
                node_ids = []
                edge_ids = []
                for j in range(n_nodes[i]):
                    node_ids.append(input_ids_i[start_id + j*(n_nodes[i] + 1)])
                    start_mask_id = start_id + j*(n_nodes[i]+1) + 1
                    edge_ids.append(input_ids_i[start_mask_id:start_mask_id+n_nodes[i]][random_order])
                node_ids = np.array(node_ids)[random_order]
                # print('node_ids', node_ids)
                edge_ids = np.array(edge_ids)[random_order]
                # print('edge_ids', edge_ids)
                input_ids_i[graph_start_id:graph_start_id+n_nodes[i]] = torch.from_numpy(node_ids)
                for j in range(n_nodes[i]):
                    input_ids_i[start_id + j*(n_nodes[i]+1)] = node_ids[j]
                    start_mask_id = start_id + j*(n_nodes[i]+1) + 1
                    input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = torch.from_numpy(edge_ids[j])
                labels_i = input_ids_i
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100)
            
        # input_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        # label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        # print('after_data_augment')
        # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
        # input()

        if self.masked_method == 'mix':
            p = np.random.rand()
            if p < 0.5:
                cur_masked_method = 'random'
            else:
                cur_masked_method = 'edge-wise'
        elif self.masked_method == 'all-mix':
            p = np.random.rand()
            if p < 0.2:
                cur_masked_method = 'consecutive-random'
            elif p < 0.4:
                cur_masked_method = 'edge-wise'
            elif p < 0.6:
                cur_masked_method = 'full-connection'
            elif p < 0.8:
                cur_masked_method = 'full-graph'
            else:
                cur_masked_method = 'full-graph-no-duty'
        elif self.masked_method == 'mix-graph':
            p = np.random.rand()
            if p < 0.33:
                cur_masked_method = 'full-connection'
            elif p < 0.66:
                cur_masked_method = 'full-graph'
            else:
                cur_masked_method = 'full-graph-no-duty'
        else:
            cur_masked_method = self.masked_method
        print_or_not = False
        batch_size = len(input_ids)

        if cur_masked_method == "random":
            raise ValueError('random masked method is not supported')
        elif cur_masked_method == "edge-wise":
            # raise ValueError('edge-wise masked method is not supported')
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                mask_id = 0
                input_ids_i = input_ids[i]
                labels_i = input_ids_i.clone()
                input_ids_mask = torch.zeros_like(input_ids_i, dtype=torch.bool)
                if self.data_order == 'duty vertex edge':
                    start_id = 13 + n_nodes[i] + 3 + 1 + 1
                elif self.data_order == 'vertex edge duty':
                    start_id = 4 + n_nodes[i] + 1 + 3
                node_mask = np.zeros((n_nodes[i]+1,))
                node_mask[:int( (n_nodes[i]+1) * self.noise_density)] = 1
                np.random.shuffle(node_mask)
                for j in range(n_nodes[i]+ 1):
                    if j == 0:
                        if node_mask[j] == 1:
                            if self.data_order == 'duty vertex edge':
                                # input_ids_i[3:8] = self.duty_cycle_mask_token_id
                                input_ids_i[3:8] = torch.tensor(self.sentinel_ids[mask_id:mask_id+5])
                                mask_id += 5
                                input_ids_mask[3:8] = 1
                            elif self.data_order == 'vertex edge duty':
                                input_ids_i[input_ids_i.shape[0]-7, input_ids_i.shape[0]-2] = self.duty_cycle_mask_token_id
                                input_ids_mask[input_ids_i.shape[0]-7, input_ids_i.shape[0]-2] = 1
                    else:
                        if node_mask[j] == 1:
                            start_mask_id = start_id + (j-1)*(n_nodes[i]+1)
                            # input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = self.edge_mask_token_id
                            input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = torch.tensor(self.sentinel_ids[mask_id:mask_id+n_nodes[i]])
                            mask_id += n_nodes[i]
                            input_ids_mask[start_mask_id:start_mask_id+n_nodes[i]] = 1
                if self.llm == 'flan-t5-encoder':
                    labels_i[~input_ids_mask] = -100
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100)
        
        elif cur_masked_method == "full-connection":
            # print('input_ids', input_ids)
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = input_ids[i]
                labels_i = input_ids_i.clone()
                input_ids_mask = torch.zeros_like(input_ids_i, dtype=torch.bool)
                if self.data_order == 'duty vertex edge':
                    input_ids_i[3:8] = self.duty_cycle_mask_token_id
                    input_ids_mask[3:8] = 1
                    start_id = 13 + n_nodes[i] + 3 + 1 + 1
                elif self.data_order == 'vertex edge duty':
                    start_id = 4 + n_nodes[i] + 1 + 3
                    input_ids_i[input_ids_i.shape[0]-7, input_ids_i.shape[0]-2] = self.duty_cycle_mask_token_id
                    input_ids_mask[input_ids_i.shape[0]-7, input_ids_i.shape[0]-2] = 1
                node_mask = np.ones((n_nodes[i],), dtype=np.int8)
                for j in range(n_nodes[i]):
                    if node_mask[j] == 1:
                        start_mask_id = start_id + (j)*(n_nodes[i]+1)
                        input_ids_i[start_mask_id-1] = self.node_mask_token_id
                        input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = self.edge_mask_token_id
                        input_ids_mask[start_mask_id-1] = 1
                        input_ids_mask[start_mask_id:start_mask_id+n_nodes[i]] = 1
                if self.llm == 'flan-t5-encoder':
                    labels_i[~input_ids_mask] = -100
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100)
        elif cur_masked_method == "full-connection-no-duty":
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = input_ids[i]
                labels_i = input_ids_i.clone()
                input_ids_mask = torch.zeros_like(input_ids_i, dtype=torch.bool)
                if self.data_order == 'duty vertex edge':
                    start_id = 13 + n_nodes[i] + 3 + 1 + 1
                elif self.data_order == 'vertex edge duty':
                    start_id = 4 + n_nodes[i] + 1 + 3
                node_mask = np.ones((n_nodes[i],), dtype=np.int8)
                for j in range(n_nodes[i]):
                    if node_mask[j] == 1:
                        start_mask_id = start_id + (j)*(n_nodes[i]+1)
                        input_ids_i[start_mask_id-1] = self.node_mask_token_id
                        input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = self.edge_mask_token_id
                        input_ids_mask[start_mask_id-1] = 1
                        input_ids_mask[start_mask_id:start_mask_id+n_nodes[i]] = 1
                if self.llm == 'flan-t5-encoder':
                    labels_i[~input_ids_mask] = -100
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=-100)
        elif cur_masked_method == "full-graph":
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = input_ids[i]
                labels_i = input_ids_i.clone()
                if self.data_order == 'duty vertex edge':
                    input_ids_i[3:8] = self.duty_cycle_mask_token_id
                    input_ids_i[13:13+n_nodes[i]] = self.node_mask_token_id
                    start_id = 13 + n_nodes[i] + 3 + 1
                elif self.data_order == 'vertex edge duty':
                    input_ids_i[input_ids_i.shape[0]-7, input_ids_i.shape[0]-2] = self.duty_cycle_mask_token_id
                    input_ids_i[4:4+n_nodes[i]] = self.node_mask_token_id
                    start_id = 4 + n_nodes[i] + 1 + 3
                else:
                    raise ValueError('data_order is not supported')
                node_mask = np.ones((n_nodes[i],), dtype=np.int8)
                for j in range(n_nodes[i]):
                    if node_mask[j] == 1:
                        start_mask_id = start_id + (j)*(n_nodes[i]+1)
                        input_ids_i[start_mask_id-1] = self.node_mask_token_id
                        input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = self.edge_mask_token_id
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        elif cur_masked_method == "full-graph-no-duty":
            input_ids_list = []
            labels_list = []
            for i in range(batch_size):
                input_ids_i = input_ids[i]
                labels_i = input_ids_i.clone()
                input_ids_mask = torch.zeros_like(input_ids_i, dtype=torch.bool)
                if self.data_order == 'duty vertex edge':
                    # input_ids_i[3:8] = self.duty_cycle_mask_token_id
                    input_ids_i[13:13+n_nodes[i]] = self.node_mask_token_id
                    start_id = 13 + n_nodes[i] + 3 + 1
                elif self.data_order == 'vertex edge duty':
                    # input_ids_i[input_ids_i.shape[0]-7, input_ids_i.shape[0]-2] = self.duty_cycle_mask_token_id
                    input_ids_i[4:4+n_nodes[i]] = self.node_mask_token_id
                    start_id = 4 + n_nodes[i] + 1 + 3
                else:
                    raise ValueError('data_order is not supported')
                node_mask = np.ones((n_nodes[i],), dtype=np.int8)
                for j in range(n_nodes[i]):
                    if node_mask[j] == 1:
                        start_mask_id = start_id + (j)*(n_nodes[i]+1)
                        input_ids_i[start_mask_id-1] = self.node_mask_token_id
                        input_ids_i[start_mask_id:start_mask_id+n_nodes[i]] = self.edge_mask_token_id
                input_ids_list.append(input_ids_i)
                labels_list.append(labels_i)
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                labels_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        else:
            raise NotImplementedError

        # input_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
        # label_str = self.tokenizer.batch_decode(labels, skip_special_tokens=False)
        # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
        # assert(labels.shape == input_ids.shape)
        # # print('input_ids', input_ids.shape, ',  labels', labels.shape)
        # # print('labels', labels)
        # input()



        batch["labels"] = labels
        batch["input_ids"] = input_ids
        # decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
        # batch["decoder_input_ids"] = decoder_input_ids

        # batch["decoder_input_ids"] = torch.tensor(
        #     shift_tokens_right(
        #         batch["labels"], self.pad_token_id, self.decoder_start_token_id
        #     )
        # )

        return batch


def process_batch_remove_eos(batch_input_ids, batch_label_ids, eos_token_id, n, max_length=None):
    batch_size = len(batch_input_ids)
    new_input_ids_batch = []
    new_label_ids_batch = []

    for i in range(batch_size):
        # Remove the EOS token from the end of input_ids and label_ids
        input_ids = batch_input_ids[i][:-1] if batch_input_ids[i][-1] == eos_token_id else batch_input_ids[i]
        label_ids = batch_label_ids[i][:-1] if batch_label_ids[i][-1] == eos_token_id else batch_label_ids[i]
        
        # Directly concatenate input_ids and label_ids for each example
        combined_sequence = torch.cat([input_ids, label_ids])
        
        # Determine the random split point
        # max_split_length = combined_sequence.size(0) - 1
        # n = np.random.randint(1, max_split_length) if max_length is None else min(max_length, max_split_length)
        
        # Split the combined sequence
        new_input_ids = combined_sequence[:n]
        new_label_ids = combined_sequence[n:]
        
        # Append the EOS token to the end of new_input_ids and new_label_ids if not empty
        new_input_ids = torch.cat([new_input_ids, eos_token_id.unsqueeze(0)], dim=0)
        new_label_ids = torch.cat([new_label_ids, eos_token_id.unsqueeze(0)], dim=0) if new_label_ids.size(0) > 0 else new_label_ids
        # print('new_input_ids', new_input_ids.size(), 'new_label_ids', new_label_ids.size())

        new_input_ids_batch.append(new_input_ids)  # Add batch dimension
        new_label_ids_batch.append(new_label_ids)  # Add batch dimension

    # Concatenate all the processed sequences back into batch tensors
    # new_input_ids_batch = torch.cat(new_input_ids_batch, dim=0)
    # new_label_ids_batch = torch.cat(new_label_ids_batch, dim=0)

    return new_input_ids_batch, new_label_ids_batch

@dataclass
class DataCollatorForCondGen:
    tokenizer: PreTrainedTokenizerBase
    data_augment: bool = False
    baseline_format: str = 'shrink_canonical'
    random_causal: bool = False
    duty_ten: bool = False
    use_duty_cycle_option_prefix: bool = True
    typeNidx: bool = False
    output_no_type: bool = False
    common_word: bool = False
    matrix_half: bool = False

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        input_ids, labels, prefixes = tuple([instance[key] for instance in examples] for key in ("input_ids", "labels", "prefixes"))

        if self.data_augment:
            # print('before data augment')
            # # print(labels)
            # org_input_str = self.tokenizer.batch_decode(input_ids)
            # org_label_str = self.tokenizer.batch_decode(labels)
            # print('input_str', '\n', org_input_str, '\nlabel_str\n', org_label_str)
            if self.baseline_format == 'shrink_canonical':
                if not self.typeNidx:
                    if not self.duty_ten:
                        if type(self.tokenizer) == T5Tokenizer:
                            # print('T5Tokenizer')
                            special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 32132])
                            edge_sep_token_id = [3, 6]
                        else:
                            special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 8])
                            edge_sep_token_id = [9]
                    else:
                        special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 3])
                        edge_sep_token_id = [4]
                    np.random.seed(None)
                    input_ids_list = []
                    labels_list = []
                    for i in range(len(input_ids)): 
                        input_ids_i = input_ids[i].tolist()
                        labels_ids_i = labels[i].tolist()
                        # print('before augment')
                        # print('input_ids_i', input_ids_i)
                        # print('labels_ids_i', labels_ids_i)
                        # The data format is here
                        # Input_ids: VIN VOUT GND Sa0 Sb0 Sb1 C0 L0 <sep> </s>
                        # Labels: <duty_0.9> <sep> VIN Sb0 , VOUT C0 , GND Sb1 , Sa0 L0 , Sa0 Sb1 , Sb0 C0 L0 <sep> </s>
                        # Help me to write the code about data augmentation with node order permutation

                        node_token_ids = [token_id for token_id in input_ids_i if token_id not in special_token_ids]
                        node_indices = {token_id: index for index, token_id in enumerate(node_token_ids)}

                        # Generate a permutation of the node indices
                        permuted_indices = list(node_indices.values())
                        permuted_indices = np.random.permutation(len(node_token_ids))

                        # Apply permutation to input token IDs
                        permuted_node_token_ids = [node_token_ids[i] for i in permuted_indices]
                        permuted_input_token_ids = permuted_node_token_ids + [token_id for token_id in input_ids_i if token_id in special_token_ids]

                        netlist_ids = labels_ids_i[2:-2]
                        edge_list = []
                        edge = []
                        for netlist_id in netlist_ids:
                            if netlist_id is edge_sep_token_id[0]:
                                edge_list.append(edge)
                                # print(edge_list)
                                edge = []
                            elif len(edge_sep_token_id) > 1 and netlist_id is edge_sep_token_id[1]:
                                continue
                            else:
                                edge.append(netlist_id)
                        edge_list.append(edge)
                        # print(edge_list)
                        node_indices = {token_id: index for index, token_id in enumerate(permuted_node_token_ids)}
                        for j in range(len(edge_list)):
                            edge_list[j].sort(key=lambda val: node_indices[val])
                        edge_list.sort(key=lambda val: node_indices[val[0]])
                        new_netlist_ids = []
                        for j, edge in enumerate(edge_list):
                            new_netlist_ids += edge
                            if j != len(edge_list) - 1:
                                new_netlist_ids.extend(edge_sep_token_id)
                        # print('new_netlist_ids', new_netlist_ids)
                        input_ids_i = torch.tensor(permuted_input_token_ids)
                        labels_ids_i = torch.tensor(labels_ids_i[:2] + new_netlist_ids + labels_ids_i[-2:])
                        # print('after augment')
                        # print('input_ids_i', input_ids_i)
                        # print('labels_ids_i', labels_ids_i)
                        # input()
                        input_ids_list.append(input_ids_i)
                        labels_list.append(labels_ids_i)
                    input_ids = input_ids_list 
                    labels = labels_list
                else:
                    # input:  VIN VOUT GND Sa 0 Sb 1 Sb 2 Sb 3 C 4 <sep> </s>
                    # label:  <duty_0.9> <sep> VIN Sb 2 , VOUT C 4 , GND Sa 0 , Sa 0 Sb 3 , Sb 1 Sb 3 , Sb 1 Sb 2 C 4 <sep> </s>
                    np.random.seed(None)
                    input_ids_list = []
                    labels_list = []
                    node_nums = []
                    input_strs = self.tokenizer.batch_decode(input_ids)
                    label_strs = self.tokenizer.batch_decode(labels)
                    for i in range(len(input_strs)):
                        
                        # replace
                        if type(self.tokenizer) == T5Tokenizer:
                            input_str = input_strs[i].split(' ')[:-2]
                            label_str = label_strs[i].replace(',', ' ,').split(' ')[:-2]
                        else:
                            input_str = input_strs[i].split(' ')[:-1]
                            label_str = label_strs[i].split(' ')[:-1]
                        # print('label_str', label_str)
                        # label_str = label_strs[i].split(' ')[:-2]
                        netlist_str = label_str[2:]
                        duty_str = label_str[:2]
                        node_list = []
                        node_id_map = {}
                        id_node_map = {}
                        if self.output_no_type == False:
                            idx = -3
                        else:
                            idx = 0
                        j = 0
                        while j < len(input_str):
                            node = input_str[j]
                            if self.output_no_type == False and self.common_word == False and (node == "VIN" or node == "VOUT" or node == "GND") or self.common_word == True and (node == 'A' or node == 'B' or node == 'C'):
                                node_list.append(idx)
                                node_id_map[node] = idx
                                id_node_map[idx] = node
                                idx += 1
                            else:
                                if j == len(input_str) - 1:
                                    break
                                node = node + ' ' + input_str[j+1]
                                node_list.append(idx)
                                node_id_map[node] = idx
                                id_node_map[idx] = node
                                idx += 1
                                j+=1
                            j+=1
                        node_nums.append(len(node_list))
                        edge_list = []
                        edge = []
                        # print('node_list', node_list)
                        # print('id_node_map', id_node_map)
                        # print('netlist_str', netlist_str)
                        j = 0 
                        while j < len(netlist_str):
                            node = netlist_str[j]
                            if node == ',':
                                edge_list.append(edge)
                                edge = []
                            elif self.output_no_type == False and self.common_word == False and (node == "VIN" or node == "VOUT" or node == "GND") or self.common_word == True and (node == 'A' or node == 'B' or node == 'C'):
                                edge.append(node)
                            else:
                                if self.output_no_type == False:
                                    if j == len(netlist_str) - 1:
                                        break
                                    node = node + ' ' + netlist_str[j+1]
                                    edge.append(node)
                                    j += 1
                                else:
                                    node = id_node_map[int(node)]
                                    edge.append(node)
                            j += 1
                        edge_list.append(edge)
                        # print('edge_list', edge_list)
                        permuted_indices = np.random.permutation(len(node_list))
                        permuted_node_token = [node_list[i] for i in permuted_indices]
                        input_str = ''
                        random_idx = np.random.permutation(np.arange(0, 13))[:len(node_list)]
                        node_indices = {}
                        for index, idx in enumerate(permuted_node_token):
                            if self.output_no_type == False:
                                input_str += (id_node_map[idx] + ' ')
                            else:
                                node_str_list = id_node_map[idx].split(' ')
                                input_str += (node_str_list[0] + ' ' + str(random_idx[int(node_str_list[1])]) + ' ')
                            node_indices[id_node_map[idx]] = index
                        input_str += ' <sep> '
                        for j in range(len(edge_list)):
                            edge_list[j].sort(key=lambda val: node_indices[val])
                        edge_list.sort(key=lambda val: node_indices[val[0]])
                        new_netlist_str = ''
                        for j, edge in enumerate(edge_list):
                            if self.output_no_type == False:
                                new_netlist_str += ' '.join(edge)
                            else:
                                new_netlist_str += ' '.join([ str(random_idx[int(node.split(' ')[1])]) for node in edge])
                            if j != len(edge_list) - 1:
                                new_netlist_str += ' , '
                        label_str = ' '.join(duty_str) + ' ' + new_netlist_str + ' <sep> '
                        
                        input_str = " ".join(input_str.split())
                        label_str = " ".join(label_str.split())
                        
                        # print('### input_str', input_str, 'label_str', label_str)
                        input_ids_i = self.tokenizer.encode(input_str)
                        label_ids_i = self.tokenizer.encode(label_str)

                        if type(self.tokenizer) != T5Tokenizer:
                            # concatenate input and label because now we are using GPT2 for casual LM
                            input_ids_i_temp = [tid for tid in input_ids_i if tid != 220]
                            label_ids_i_temp = [tid for tid in label_ids_i if tid != 220]
                            label_ids_i = [-100] * (len(input_ids_i_temp)+7) + label_ids_i_temp + [self.tokenizer.eos_token_id]
                            input_ids_i = input_ids_i_temp + label_ids_i_temp + [self.tokenizer.eos_token_id]
                            
                        # print('input_ids_i', input_ids_i, '\nlabel_ids_i', label_ids_i)
                        input_ids_i = torch.tensor(input_ids_i)
                        label_ids_i = torch.tensor(label_ids_i)
                        input_ids_list.append(input_ids_i)
                        labels_list.append(label_ids_i)
                    input_ids = input_ids_list 
                    labels = labels_list

                    # input_ids_list = []
                    # labels_list = []
                    # np.random.seed(None)
                    # number_id_set = set([i for i in range(21, 34)])
                    # for i in range(len(input_ids)): 
                    #     input_ids_i = input_ids[i].tolist()
                    #     label_ids_i = labels[i].tolist()
                    #     node_num = node_nums[i]

                    #     random_idx = np.random.permutation(np.arange(21, 34))[:node_num]
                    #     idx = 0
                    #     original2random_idx_map = {}
                    #     for j, token_id in enumerate(input_ids_i):
                    #         if token_id in number_id_set:
                    #             original2random_idx_map[token_id] = random_idx[idx]
                    #             input_ids_i[j] = random_idx[idx]
                    #             idx += 1
                    #     for j, token_id in enumerate(label_ids_i):
                    #         if token_id in number_id_set:
                    #             label_ids_i[j] = original2random_idx_map[token_id]
                    #     input_ids_i = torch.tensor(input_ids_i)
                    #     label_ids_i = torch.tensor(label_ids_i)
                    #     input_ids_list.append(input_ids_i)
                    #     labels_list.append(label_ids_i)
                    # input_ids = input_ids_list 
                    # labels = labels_list

            elif self.baseline_format == 'matrix':
                np.random.seed(None)
                input_ids_list = []
                labels_list = []
                if self.tokenizer.sep_token_id == None:
                    special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 3, 4])
                else:
                    special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id])
                for i in range(len(input_ids)):
                    # input_ids_i = input_ids[i].numpy()
                    # label_ids_i = labels[i].numpy()
                    input_ids_i = np.array(input_ids[i])
                    label_ids_i = np.array(labels[i])
                    # input:  VIN VOUT GND Sa Sa Sb L L <sep> </s>
                    # label:  <duty_0.3> <sep> VIN <no_edge> <no_edge> <no_edge> <no_edge> <no_edge> <edge_1> <no_edge> <no_edge> VOUT <no_edge> <no_edge> <no_edge> <edge_1> <no_edge> <no_edge> <no_edge> <edge_1> GND <no_edge> <no_edge>
                    # <no_edge> <no_edge> <no_edge> <no_edge> <edge_1> <no_edge> Sa <no_edge> <edge_1> <no_edge> <no_edge> <edge_2> <no_edge> <no_edge> <both_edges> Sa <no_edge> <no_edge> <no_edge> <edge_1> <no_edge> <edge_2> <edge_2> <e
                    # dge_1> Sb <edge_1> <no_edge> <no_edge> <no_edge> <edge_2> <no_edge> <edge_2> <no_edge> L <no_edge> <no_edge> <edge_1> <no_edge> <edge_2> <edge_2> <no_edge> <no_edge> L <no_edge> <edge_1> <no_edge> <both_edges> <edge
                    # _2> <no_edge> <no_edge> <no_edge> <sep> </s>
                    node_token_ids = [token_id for token_id in input_ids_i if token_id not in special_token_ids]
                    node_indices = {token_id: index for index, token_id in enumerate(node_token_ids)}

                    # Generate a permutation of the node indices
                    permuted_indices = list(node_indices.values())
                    permuted_indices = np.random.permutation(len(node_token_ids))
                    # Apply permutation to input token IDs
                    permuted_node_token_ids = [node_token_ids[i] for i in permuted_indices]
                    permuted_input_token_ids = permuted_node_token_ids+ [token_id for token_id in input_ids_i if token_id in special_token_ids]

                    n_node = len(permuted_node_token_ids)
                    netlist_ids = label_ids_i[2:-2]
                    edge_ids = []
                    for j in range(len(permuted_node_token_ids)):
                        start_mask_id = j * (n_node + 1) + 1
                        edge_ids.append(netlist_ids[start_mask_id:start_mask_id+n_node][permuted_indices])
                    edge_ids = np.array(edge_ids)[permuted_indices]
                    # print('len of edge_ids', edge_ids.shape)
                    input_ids_i = torch.tensor(permuted_input_token_ids)
                    
                    for j in range(n_node):
                        netlist_ids[j*(n_node+1)] = permuted_node_token_ids[j]
                        start_mask_id = j * (n_node + 1) + 1
                        netlist_ids[start_mask_id:start_mask_id+n_node] = edge_ids[j]
                    label_ids_i = torch.tensor(label_ids_i[:2].tolist() + netlist_ids.tolist() + label_ids_i[-2:].tolist())
                    input_ids_list.append(input_ids_i)
                    labels_list.append(label_ids_i)
                input_ids = input_ids_list 
                labels = labels_list
            else:
                raise ValueError('baseline_format should be shrink_canonical or matrix')
        
        if self.matrix_half:
            np.random.seed(None)
            input_ids_list = []
            labels_list = []
            if self.tokenizer.sep_token_id == None:
                special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 3, 4])
            else:
                special_token_ids = set([self.tokenizer.bos_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, self.tokenizer.sep_token_id])
            input_strs = self.tokenizer.batch_decode(input_ids)
            label_strs = self.tokenizer.batch_decode(labels)
            # edgeId_2_str = {'<edge_1>': 0, '<edge_2>': 1} 
            # edge_str = ['<edge_1>', '<edge_2>']
            # : '<no_edge>',  2: '<both_edges>'
            for i in range(len(input_strs)):
                input_str = input_strs[i].split(' ')[:-2]
                n_node = len(input_str)
                # replace
                label_str = label_strs[i].split(' ')[:-2]
                # print('input_str', input_str)
                # print('label_str', label_str)
                # label_str = label_strs[i].split(' ')[:-2]
                netlist_str = label_str[2:]
                duty_str = label_str[:2]
                new_netlist_str = []
                j = 0
                n_idx = 0
                while j < len(netlist_str):
                    node = netlist_str[j]
                    new_netlist_str.append(node)
                    ignore_set = set()
                    edge_id =0
                    edge_id_dict = {}
                    j += 1
                    for k in range(n_node):
                        # print('netlist_str[j]', netlist_str[j])
                        if k <= n_idx:
                            if netlist_str[j] != '<no_edge>':
                                ignore_set.add(netlist_str[j])
                        else:
                            ignore_set_size = len(ignore_set)
                            if netlist_str[j] != '<no_edge>':
                                if netlist_str[j] not in ignore_set:
                                    if netlist_str[j] == '<both_edges>':
                                        new_netlist_str.append('<both_edges>')
                                    else:
                                        if '<edge_1>' in ignore_set and '<edge_2>' == netlist_str[j]:
                                            new_netlist_str.append('<edge_1>')
                                        else:
                                            new_netlist_str.append(netlist_str[j])
                                        # new_netlist_str.append( edge_str[edgeId_2_str[netlist_str[j]] - ignore_set_size] )
                                else:
                                    new_netlist_str.append('<no_edge>')
                            else:
                                new_netlist_str.append('<no_edge>')
                        j+=1
                    # print(n_idx, len(new_netlist_str))
                    n_idx += 1
                input_str = ' '.join(input_str) + ' <sep> '
                label_str = ' '.join(duty_str) + ' ' + ' '.join(new_netlist_str) + ' <sep> '
                # print('after augment input_str', input_str)
                # print('after augment label_str', label_str)
                input_ids_i = self.tokenizer.encode(input_str)
                label_ids_i = self.tokenizer.encode(label_str)
                # print('input_ids_i', input_ids_i, 'label_ids_i', label_ids_i)
                input_ids_i = torch.tensor(input_ids_i)
                label_ids_i = torch.tensor(label_ids_i)
                input_ids_list.append(input_ids_i)
                labels_list.append(label_ids_i)
            input_ids = input_ids_list 
            labels = labels_list

        if self.random_causal:
            min_length = 1000000
            for i in range(len(input_ids)):
                input_ids_i = input_ids[i]
                label_ids_i = labels[i]
                length = input_ids_i.size(0) + label_ids_i.size(0)
                if length < min_length:
                    min_length = length
            n = np.random.randint(1, min_length - 2)
            input_ids, labels = process_batch_remove_eos(input_ids, labels, torch.tensor(self.tokenizer.eos_token_id), n)
            # eos_token_id = torch.tensor(2) 
            # print('before random_causal')
            # in this section, we would like to randomly add n label token to input_ids
            # and remove n token from the end of the input_ids

        if not self.data_augment and type(self.tokenizer) != T5Tokenizer:
            # concatenate input and label because now we are using GPT2 for casual LM
            # print('GPT2')
            input_ids_list = []
            labels_list = []
            for i in range(len(input_ids)):
                input_ids_i = input_ids[i].tolist()
                labels_ids_i = labels[i].tolist()
                input_ids_i_temp = [tid for tid in input_ids_i if tid != 220]
                label_ids_i_temp = [tid for tid in labels_ids_i if tid != 220]
                labels_ids_i = [-100] * (len(input_ids_i_temp)+7) + label_ids_i_temp + [self.tokenizer.eos_token_id]
                input_ids_i = input_ids_i_temp + label_ids_i_temp + [self.tokenizer.eos_token_id]
                # print('input_ids_i', input_ids_i)
                # print('labels_ids_i', labels_ids_i)
                input_ids_list.append(torch.tensor(input_ids_i))
                labels_list.append(torch.tensor(labels_ids_i))
            input_ids = input_ids_list 
            labels = labels_list

        batch = dict()
        batch['vout'] = torch.stack([prefixes[i]['vout'] for i in range(len(prefixes))])
        batch['eff'] = torch.stack([prefixes[i]['eff'] for i in range(len(prefixes))])
        if self.use_duty_cycle_option_prefix:
            batch['d_cycle_option'] = torch.stack([prefixes[i]['d_cycle_option'] for i in range(len(prefixes))])
        if type(self.tokenizer) == T5Tokenizer:
            batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        else:
            # GPT2 tokenizer
            batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        # print('input_ids', batch['input_ids'].size())
        batch['labels'] = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print('labels', batch['labels'].size())
        
        # print('after_data_augment')
        # # # print('label', batch['labels'])
        # input_str = self.tokenizer.batch_decode(batch['input_ids'])
        # label_str = self.tokenizer.batch_decode(batch['labels'])
        
        # print('input_str', '\n', input_str, '\nlabel_str\n', label_str)
        # input()

        # print('vout', batch['vout'])
        # print('eff', batch['eff'])
        # if self.use_duty_cycle_option_prefix:
        #     print('d_cycle_option', batch['d_cycle_option'])
        
        # # netlist, duty = read_transformer_output_mask(org_input_str[0], org_label_str[0], duty10=False, pre_eval=True)
        # # print('org netlist', netlist)
        # # print('org duty', duty)
        # netlist, duty_cycle = read_transformer_output_mask(input_str[0], label_str[0], duty10=False, pre_eval=True)
        # result = sim_netlist_duty_cycle('try.cki', netlist, duty_cycle)
        # print(result)
        # print('netlist', netlist)
        # print('duty', duty_cycle)
        
        
        # print('==== in DataCollatorForCondGen ', batch['input_ids'].size(), batch['labels'].size())
        # input('pause')
        return batch