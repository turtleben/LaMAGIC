import os
import sys
import logging

import torch
import transformers
import json
from torch.utils.data import DataLoader, ConcatDataset


from analog_LLM.utils.utils import *

from analog_LLM.utils.prompter import Prompter
from analog_LLM.utils.dataset import *
from analog_LLM.utils.data_collator import DataCollatorForT5MLM, compute_input_and_target_lengths, DataCollatorForGraphMLM
from analog_LLM.models.T5_prefix import T5ForConditionalGeneration, T5EncoderModel

class AnalogLLMBuilder():
    def __init__(self, parameters):
        self.params = parameters
        self.params.print()
        
        logging.warning("_init_pretrained_model ...")
        print('##### _init_pretrained_model')
        self._init_pretrained_model()
        self.stat_dict = None
        if not self.params.val_custom:
            print('##### _init_dataset')
            self._init_dataset()
        
    def _init_dataset(self):
        if self.params.task == 'maskedGen':
            if not self.params.tokenized:
                d_path = os.path.join(self.params.text_data_dir, self.params.target_data)
                data = json.load(open(d_path, 'r'))
                print('total data', len(data))
                self.data_trn, self.data_val = random_split_trn_val(self.params, data, self.params.val_set_size)
                print('number of training data', len(self.data_trn), '\nnumber of testing data', len(self.data_val))
                os.makedirs(self.params.tokenized_data_dir, exist_ok=True)
                d_path_trn = os.path.join(self.params.tokenized_data_dir, self.params.tokenized_data_trn)
                d_path_val = os.path.join(self.params.tokenized_data_dir, self.params.tokenized_data_val)
                self.dset_trn = RawDatasetMaskedGen(self.params, self.data_trn, self.tokenizer, save_tokenized_path=d_path_trn)
                self.dset_val = RawDatasetMaskedGen(self.params, self.data_val, self.tokenizer, save_tokenized_path=d_path_val)
                return
            else:
                d_dict_trn = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data_trn, False)
                d_dict_val = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data_val, False)
                d_set = NormalizeDatasetMaskGen(self.params, d_dict_trn, d_dict_val)
                self.stat_dict = d_set.return_statistic()
                print(self.stat_dict)
                self.dset_trn = TokenizedDatasetMaskGen(self.params, d_dict_trn, self.stat_dict, data_num=self.params.trn_data_num)
                print('##### dset_trn', len(self.dset_trn))
                self.dset_val = TokenizedDatasetMaskGen(self.params, d_dict_val, self.stat_dict)
                print('##### dset_val', len(self.dset_val))
                _, self.targets_length = compute_input_and_target_lengths(
                                                                inputs_length=self.params.cutoff_len,
                                                                noise_density=self.params.masked_ratio,
                                                                mean_noise_span_length=3,
                                                            )
                print('n_new_tokens', self.params.n_new_tokens)
                if self.params.mask_style == 'graph_mask':
                    self.data_collator = DataCollatorForGraphMLM(
                        tokenizer=self.tokenizer,
                        masked_method=self.params.masked_method,
                        data_order=self.params.order,
                        noise_density=self.params.masked_ratio,
                        mean_noise_span_length=6,
                        input_length=self.params.cutoff_len,
                        target_length=self.targets_length,
                        pad_token_id=self.model.config.pad_token_id,
                        decoder_start_token_id=self.model.config.decoder_start_token_id,
                        n_new_tokens=self.params.n_new_tokens,
                        data_augment=self.params.data_augment,
                        llm=self.params.llm,
                    )
                else:
                    self.data_collator = DataCollatorForT5MLM(
                            tokenizer=self.tokenizer,
                            masked_method=self.params.masked_method,
                            data_order=self.params.order,
                            noise_density=self.params.masked_ratio,
                            mean_noise_span_length=6,
                            input_length=self.params.cutoff_len,
                            target_length=self.targets_length,
                            pad_token_id=self.model.config.pad_token_id,
                            decoder_start_token_id=self.model.config.decoder_start_token_id,
                            n_new_tokens=self.params.n_new_tokens,
                            data_augment=self.params.data_augment,
                            llm = self.params.llm,
                        )
                return
        tokenized=self.params.tokenized
        if not tokenized:
            print(f'##### not tokenized, in {self.params.task}')
            d_path = os.path.join(self.params.text_data_dir, self.params.target_data)
            data = json.load(open(d_path, 'r'))
            self.data_trn, self.data_val = random_split_trn_val(self.params, data, self.params.val_set_size)
            print('number of training data', len(self.data_trn), '\nnumber of testing data', len(self.data_val))
            os.makedirs(self.params.tokenized_data_dir, exist_ok=True)
            d_path_trn = os.path.join(self.params.tokenized_data_dir, self.params.tokenized_data_trn)
            d_path_val = os.path.join(self.params.tokenized_data_dir, self.params.tokenized_data_val)
            # Tokenized the text dataset and save it
            # self.data = load_data(self.params)
            # val_d_num = int(len(self.data["train"])*self.params.val_set_size)
            if self.params.task == 'causal':
                self.dset_trn = SupervisedDataset(self.params, self.data["train"], self.tokenizer)
            elif self.params.task == 'regression':
                self.dset_trn = RawTextDatasetRegression(self.params, self.data["train"], self.tokenizer)
            elif self.params.task == 'conditionalGen':
                self.dset_trn = RawTextDatasetConditionalGen(self.params, self.data_trn, self.tokenizer, save_tokenized_path=d_path_trn)
                self.dset_val = RawTextDatasetConditionalGen(self.params, self.data_val, self.tokenizer, save_tokenized_path=d_path_val)
            else:
                raise NotImplementedError
            # self.dset_val = SupervisedDataset(self.val_data, self.tokenizer)
        else:
            # Initialize the dataset for training and testing
            # input_trn, input_val, label_trn, label_val = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data, True, self.params.val_set_size)
            # self.dset_trn = TokenizedDataset(self.params, input_trn, label_trn, data_num=self.params.trn_data_num)
            # self.dset_val = TokenizedDataset(self.params, input_val, label_val, data_num=self.params.trn_data_num)

            d_dict_trn = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data_trn, False)
            self.dset_trn = TokenizedDataset(self.params, d_dict_trn["input_ids"], d_dict_trn["labels"], data_num=self.params.trn_data_num, tokenizer=self.tokenizer)
            d_dict_val = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data_val, False)
            self.dset_val = TokenizedDataset(self.params, d_dict_val["input_ids"], d_dict_val["labels"], data_num='all', tokenizer=self.tokenizer)
            
            if self.params.data_augment == True:
                input_trn, input_val, label_trn, label_val = load_tokenized_data(self.params.tokenized_data_dir_augment, True, self.params.tokenized_data_augment)
                self.dset_trn_aug = TokenizedDataset(self.params, input_trn, label_trn, data_num=self.params.trn_data_num)
                # self.dset_val_aug = TokenizedDataset(input_val, label_val, data_num=self.params.trn_data_num, generate=self.params.generate)
                self.dset_trn = ConcatDataset([self.dset_trn, self.dset_trn_aug])
                print(len(self.dset_trn))
            # output = self.tokenizer.decode(self.dset_trn[0]["input_ids"])
            # print('input_ids: ', output)
            # output = self.tokenizer.decode(self.dset_trn[0]["labels"])
            # print('labels: ', output)
            # input()
        # Data collator will perform preprocessing like padding 
        self.data_collator = DataCollatorForSupervisedDataset(tokenizer=self.tokenizer, task=self.params.task)
    
    def _init_pretrained_model(self):
        # self.prompter = Prompter(self.params.prompt_template_name)
        device_map = "auto"
        self.params.gradient_accumulation_steps = self.params.batch_size // self.params.micro_batch_size
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.params.ddp = world_size != 1
        print('world_size: ', world_size)
        if self.params.ddp:
            print(os.environ.get("LOCAL_RANK"))
            device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
            self.params.gradient_accumulation_steps = self.params.gradient_accumulation_steps // world_size
        if self.params.generate or self.params.llm == 'bert' or (self.params.llm == 'flan-t5' and self.params.ddp != True):
            device_map={"":self.params.LLM_device}
        
        # set the model path to restore weights
        base_model_path = self.params.base_model
        if self.params.finetune_method == 'pure':
            base_model_path = self.params.base_model if self.params.trn_or_val == 'train' else self.params.output_dir
        if self.params.finetune_from_ours:
            base_model_path = self.params.our_model_dir if self.params.trn_or_val == 'train' else self.params.output_dir
        
        # use the user-defined parameter to get specific model, tokenizer, and some config
        load_param, llm_config, Llm_model, Llm_tokenizer = generate_llm_config(self.params, base_model_path)
        
        print('move load in base_model_path: ', base_model_path)
        # initialize the model and tokenizer 
        if llm_config != None:
            if self.params.load_pretrained == False:
                self.model = Llm_model(config=llm_config)
            else:
                self.model = Llm_model.from_pretrained(
                    base_model_path,
                    device_map=device_map,
                    config=llm_config,
                    **load_param
                )
        else:
            self.model = Llm_model.from_pretrained(
                base_model_path,
                device_map=device_map,
                **load_param
            )
        
        self.tokenizer = Llm_tokenizer.from_pretrained(
            self.params.base_model,
            model_max_length=self.params.cutoff_len,
            padding_side="right",
            use_fast=False,
            legacy=True,
        )
        if self.params.llm == 'llama':
            smart_tokenizer_and_embedding_resize(
                tokenizer=self.tokenizer,
                model=self.model,
            )
        add_device_token(args=self.params, tokenizer=self.tokenizer, model=self.model)

        if self.params.encoder_model_dir != None:
            encoder_model = T5EncoderModel.from_pretrained(self.params.encoder_model_dir,
                        device_map=device_map, config=llm_config, **load_param) 
            self.model.encoder.load_state_dict(encoder_model.encoder.state_dict())
            self.model.shared.load_state_dict(encoder_model.shared.state_dict())
            self.model.vout_linear.load_state_dict(encoder_model.vout_linear.state_dict())
            del encoder_model
            print('load encoder')
            # input()

        # self.tokenizer.pad_token_id = (0)  # unk. we want this to be different from the eos token
        # self.tokenizer.padding_side = "right"  # Allow batched inference ## follow stanford_alpaca
        
    def train(self):
        if self.params.finetune_method == 'lora':
            finetune_lora(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val)
        elif self.params.finetune_method == 'pure':
            if self.params.task == 'maskedGen':
                finetune_maskedGen(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val)
            else:
                finetune(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val)
        
    def val(self, get_mse=False, sim=False, comp6=False):
        self.params.text_data_dir = "/skunk-pod-storage-chenchia-2echang-40duke-2eedu-pvc/dataset_power_converter/text_dataset/masked"
        self.params.target_data = "dataset_all_345_regenerate_prune_isomophic_new.json"
        d_path = os.path.join(self.params.text_data_dir, self.params.target_data)
        if comp6:
            d_path = os.path.join(self.params.text_data_dir, self.params.LUT_cir_data_name)
        cir_data = json.load(open(d_path, 'r'))
        if  self.params.task == 'causal' or self.params.task == 'conditionalGen' :
            val(self.params, self.model, self.tokenizer, self.dset_trn, self.dset_val, self.data_collator, cir_data, get_mse, sim)
        elif self.params.task == 'maskedGen':
            # val(self.params, self.model, self.tokenizer, self.dset_trn, self.dset_val, get_mse, sim)
            if self.params.masked_method == 'regression':
                val_maskedRegression(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val, self.stat_dict, get_mse)
            else:
                if not self.params.val_custom:
                    self.data_collator.data_augment = False
                    val_maskedGen(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val, cir_data, self.stat_dict, get_mse, sim)
                else:
                    val_maskedGen_custom_input(self.params, self.model, self.tokenizer, self.stat_dict, cir_data)
            # val_maskedRegression(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val, get_mse)
            # Use the customized input to model
            # val_custum_input(self.params, self.model, self.tokenizer)
        elif self.params.task == 'regression':
            val_regression(self.params, self.model, self.tokenizer, self.dset_trn, self.dset_val)
            # val_maskedRegression(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val, get_mse)
        else:
            raise NotImplementedError
        
    def check_data_augment(self):
        # make sure that the prediction dataset are the same as the generation dataset
        check_data_augment(self.tokenizer, self.dset_trn, self.dset_trn_aug)
    
    def generate(self):
        pass
        
        
        
