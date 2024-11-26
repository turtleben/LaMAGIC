import os
import sys
from typing import List, Dict, Optional, Sequence


import torch
import transformers
from dataclasses import dataclass, field
from peft import LoraConfig
from transformers import LlamaConfig, BertConfig
from transformers import LlamaForCausalLM, LlamaForSequenceClassification, LlamaTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import T5Config, T5Tokenizer
from analog_LLM.utils.tokenizer import CustomTokenizer
from analog_LLM.models.T5_prefix import T5ForConditionalGeneration, T5EncoderModel, T5ForRegression

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    
def generate_llm_config(args, base_model_path, rl=False):
    if args.llm == 'llama':
        tokenizer = LlamaTokenizer
        load_param = { "load_in_8bit": True, 'torch_dtype': torch.float16,}
        if args.task == 'regression':
            llm_config = LlamaConfig(problem_type="regression",num_labels=1,)
            llm_model = LlamaForSequenceClassification
        else:
            llm_config = None
            llm_model = LlamaForCausalLM
            
    elif args.llm == 'bert':
        tokenizer = BertTokenizer
        load_param = { "load_in_8bit": False}
        if args.task == 'regression':
            llm_config = BertConfig(problem_type="regression",num_labels=1,)
            llm_model = BertForSequenceClassification
        else:
            llm_config = None
            raise NotImplementedError

    elif args.llm == "flan-t5-baseline":
        from transformers import T5ForConditionalGeneration as T5ForConditionalGeneration_org
        tokenizer = T5Tokenizer
        load_param = { "load_in_8bit": False}
        llm_config = None
        llm_config = T5Config.from_pretrained(args.base_model)
        llm_model = T5ForConditionalGeneration_org
            
    elif args.llm == "flan-t5":
        # tokenizer = T5TokenizerFast
        tokenizer = T5Tokenizer
        load_param = { "load_in_8bit": False}
        # if args.task == 'regression':
        #     llm_config = T5Config.from_pretrained(args.base_model)
        #     llm_config.problem_type = "regression"
        #     llm_config.num_labels = 1
        #     llm_config.classifier_dropout = 0
        #     llm_model = T5ForSequenceClassification
        #     raise NotImplementedError
        # else:
        llm_config = None
        llm_config = T5Config.from_pretrained(base_model_path)
        llm_config.problem_type = "regression"
        llm_config.num_labels = args.num_labels
        llm_config.classifier_dropout = 0
        print(llm_config.vocab_size)
        # llm_config.dropout_rate = args.dropout_rate
        # llm_model = AutoModelForSeq2SeqLM
        llm_model = T5ForConditionalGeneration
        if args.masked_method == 'regression':
            llm_model = T5ForRegression
        if rl:
            llm_critic = T5ForRegression
        print(llm_model)
        
    elif args.llm == "flan-t5-encoder":
        tokenizer = T5Tokenizer
        load_param = { "load_in_8bit": False}
        llm_config = None
        llm_config = T5Config.from_pretrained(args.base_model)
        # llm_config.num_layers = 18
        # llm_config.dropout_rate = 0.05
        llm_model = T5EncoderModel
    elif args.llm == "transformer-encoder-decoder":
        from analog_LLM.models.T5_transformer import T5ForConditionalGeneration as T5ForCondGen_Transformer
        tokenizer = CustomTokenizer
        load_param = { "load_in_8bit": False}
        llm_config = None
        llm_config = T5Config.from_pretrained(args.base_model)
        llm_config.dropout_rate = args.dropout_rate
        
        llm_model = T5ForCondGen_Transformer
    else: 
        raise NotImplementedError
    if rl:
        return load_param, llm_config, llm_model, llm_critic, tokenizer
    return load_param, llm_config, llm_model, tokenizer
    
def generate_config_param(args):
    if args.llm == 'llama':
        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if args.val_set_size > 0 else None,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            group_by_length=args.group_by_length,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
        )
    elif args.llm == 'bert':
        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.lr,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if args.val_set_size > 0 else None,
            save_steps=200,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            group_by_length=args.group_by_length,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
        )
    elif args.llm == "flan-t5" or args.llm == "flan-t5-baseline" or args.llm == "transformer-encoder-decoder":
        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            # lr_scheduler_type="constant",
            # gradient_checkpointing=True,
            learning_rate=args.lr,
            fp16=args.fp16,
            logging_steps=20,
            weight_decay=args.reg,
            optim="adamw_torch",
            # optim="adafactor",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if args.val_set_size > 0 else None,
            save_steps=args.eval_steps,
            output_dir=args.output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            group_by_length=args.group_by_length,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
            remove_unused_columns=False,
            logging_nan_inf_filter=False,
            # max_grad_norm=0.1
        )
    elif args.llm == "flan-t5-encoder":
        train_args = transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_epochs,
            # lr_scheduler_type="constant",
            # gradient_checkpointing=True,
            learning_rate=args.lr,
            fp16=args.fp16,
            logging_steps=20,
            weight_decay=args.reg,
            optim="adamw_torch",
            evaluation_strategy="steps" if args.val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=args.eval_steps if args.val_set_size > 0 else None,
            save_steps=args.eval_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if args.val_set_size > 0 else False,
            ddp_find_unused_parameters=False if args.ddp else None,
            group_by_length=args.group_by_length,
            report_to="wandb" if args.use_wandb else None,
            run_name=args.wandb_run_name if args.use_wandb else None,
            remove_unused_columns=False,
        )
    else:
        raise NotImplementedError

    if args.finetune_method == 'lora':
        if args.task == 'causal':
            task_type = "CAUSAL_LM" 
        elif args.task == 'regression':
            task_type = "SEQ_CLS"
        else:
            raise NotImplementedError
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=task_type,
        )
        return lora_config, train_args
    elif args.finetune_method == 'pure':
        return False, train_args
    else:
        return NotImplementedError