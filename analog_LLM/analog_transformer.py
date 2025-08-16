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
from analog_LLM.utils.data_collator import DataCollatorForT5MLM, compute_input_and_target_lengths, DataCollatorForGraphMLM, DataCollatorForCondGen
from analog_LLM.models.T5_prefix import T5ForConditionalGeneration, T5EncoderModel
from analog_LLM.models.T5_transformer import T5ForConditionalGeneration as T5ForCondGen_Transformer

class AnalogTransformerBuilder():
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
                # self.params.normalize = False
                # self.dset_val_unnorm = TokenizedDatasetMaskGen(self.params, d_dict_val, self.stat_dict)
                # self.params.normalize = True
                if self.params.normalize:
                    print('##### normalize dataset')
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
        else:
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
                if self.params.task == 'conditionalGen':
                    self.dset_trn = RawTextDatasetConditionalGen_Transformer(self.params, self.data_trn, self.tokenizer, save_tokenized_path=d_path_trn)
                    self.dset_val = RawTextDatasetConditionalGen_Transformer(self.params, self.data_val, self.tokenizer, save_tokenized_path=d_path_val)
                else:
                    raise NotImplementedError
            else:
                d_dict_trn = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data_trn, False)
                print('len of d_dict_trn', len(d_dict_trn['input_ids']))
                self.dset_trn = TokenizedDataset_Transformer(self.params, d_dict_trn, None, data_num=self.params.trn_data_num)
                d_dict_val = load_tokenized_data(self.params.tokenized_data_dir, self.params.tokenized_data_val, False)
                self.dset_val = TokenizedDataset_Transformer(self.params, d_dict_val, None)
            # Data collator will perform preprocessing like padding 
            self.data_collator = DataCollatorForCondGen(tokenizer=self.tokenizer, data_augment=self.params.data_augment, baseline_format=self.params.baseline_format, 
                                    random_causal=self.params.random_causal, duty_ten=self.params.duty10, 
                                    use_duty_cycle_option_prefix=self.params.use_duty_cycle_option_prefix, 
                                    typeNidx=self.params.typeNidx, output_no_type=self.params.output_no_type,
                                    common_word=self.params.common_word, matrix_half=self.params.matrix_half)
    
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
        if self.params.generate or self.params.llm == 'bert' or ((self.params.llm == 'flan-t5' or self.params.llm == 'gpt2-decoder-only') and self.params.ddp != True):
            device_map={"":self.params.LLM_device}
        
        # set the model path to restore weights
        base_model_path = self.params.base_model
        if self.params.finetune_method == 'pure':
            base_model_path = self.params.base_model if self.params.trn_or_val == 'train' else self.params.output_dir
        if self.params.finetune_from_ours:
            base_model_path = self.params.our_model_dir if self.params.trn_or_val == 'train' else self.params.output_dir
        if self.params.trn_or_val == 'val':
            self.params.load_pretrained = True
        
        # use the user-defined parameter to get specific model, tokenizer, and some config
        load_param, llm_config, Llm_model, Llm_tokenizer = generate_llm_config(self.params, base_model_path)
        
        # print('move load in base_model_path: ', base_model_path)
        if self.params.tokenizer == 'flanT5':
            # Llm_model = T5ForCondGen_Transformer
            Llm_tokenizer = transformers.T5Tokenizer
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
        elif self.params.tokenizer == 'gpt2':
            Llm_tokenizer = transformers.GPT2Tokenizer
            self.tokenizer = Llm_tokenizer.from_pretrained(
                self.params.base_model,
                model_max_length=self.params.cutoff_len,
                # padding_side="right",
                # use_fast=False,
                # legacy=True,
            )
        else:
            self.tokenizer = Llm_tokenizer(vocab_file=self.params.vocab_file, extra_ids=0)
        
        print('tokenizer len', len(self.tokenizer), type(self.tokenizer))
        # initialize the model and tokenizer 
        if llm_config != None:
            if self.params.load_pretrained == False and self.params.finetune_from_ours == False:
                print('##### load from scratch')
                llm_config.d_model = self.params.d_model
                llm_config.vocab_size = self.params.vocab_size
                self.model = Llm_model(config=llm_config)
            else:
                print('##### load from pretrained in ', base_model_path)
                # print('##### self.params.vocab_size', self.params.vocab_size)
                # llm_config.d_model = self.params.d_model
                if self.params.tokenizer != 'flanT5' and self.params.tokenizer != 'gpt2':
                    llm_config.vocab_size = self.params.vocab_size
                print('##### llm_config.vocab_size', llm_config.vocab_size)
                self.model = Llm_model.from_pretrained(
                    base_model_path,
                    device_map=device_map,
                    config=llm_config,
                    **load_param
                )
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            raise NotImplementedError
        if self.params.tokenizer == 'flanT5' or self.params.tokenizer == 'gpt2':
            add_device_token(args=self.params, tokenizer=self.tokenizer, model=self.model, transformers_formulation=True)
        print('len(self.tokenizer)', len(self.tokenizer))
        if self.params.finetune_from_ours == True:
            self.model = Llm_model.from_pretrained(
                base_model_path,
                device_map=device_map,
                **load_param
            )

    def train(self):
        if self.params.finetune_method == 'lora':
            finetune_lora(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val)
        elif self.params.finetune_method == 'pure':
            if self.params.task == 'maskedGen':
                finetune_maskedGen(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val)
            else:
                finetune(self.params, self.model, self.tokenizer, self.data_collator, self.dset_trn, self.dset_val)
        
    def val(self, get_mse=False, sim=False, comp6=False):
        text_data_dir = self.params.text_data_dir
        if not self.params.duty10:
            assert False, "change self.params.text_data_dir and target_data into your local path for json https://huggingface.co/datasets/turtleben/LaMAGIC-dataset/blob/main/transformed/LaMAGIC/matrix_form_345comp.json"
            print("after changing, comment assertion")
            self.params.text_data_dir = "[Your local path]"
            self.params.target_data = "matrix_form_345comp.json"
        else:
            assert False, "not implemented for duty10"
            # self.params.target_data = "dataset_345_10duty_matrix_dutycycle_first.json"
        d_path = os.path.join(self.params.text_data_dir, self.params.target_data)
        if comp6:
            assert False, "make sure you change self.params.text_data_dir in line 229 into your local path for json https://huggingface.co/datasets/turtleben/LaMAGIC-dataset/blob/main/transformed/LaMAGIC/matrix_form_6comp.json"
            print("after changing, comment assertion")
            d_path = os.path.join(self.params.text_data_dir , "matrix_form_6comp.json")
            self.params.text_data_dir = text_data_dir
        print('d_path', d_path)
        cir_data = json.load(open(d_path, 'r'))
        if  self.params.task == 'causal' or self.params.task == 'conditionalGen' :
            self.data_collator.data_augment = False
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
        
        
        
