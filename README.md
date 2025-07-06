# LaMAGIC and LaMAGIC2 at ICML 24, 25: Language-Model-based Topology Generation for Analog Integrated Circuits


## Package

[//]: # (The used package can be found in `package-list.txt`. You can use it to create a Conda environment.)

Run the following command to create the environment:
   ```bash
   conda env create -f environment.yml
```

After cloning, use git lfs pull to clone all large dataset files.

## Framework

This is the work for Analog topology generation. 
Article: LaMAGIC at ICML'24 `https://arxiv.org/pdf/2407.18269`
LaMAGIC2 at ICML'25 `https://arxiv.org/abs/2506.10235`

## Setup

After installing the package, download the model of flan-T5-base at `https://huggingface.co/google/flan-t5-base`.
This is the base model we use for finetuning.
Then, for all the yml files under directory `analog_LLM/configs`, change the base_model into your model save path.

The dataset is in `https://huggingface.co/datasets/turtleben/LaMAGIC-dataset`. You should clone this dataset first. Then, the target data is located in `[your_save_path]/LaMAGIC-dataset/transformed`. The SOTA model version is using data `LaMAGIC2/SFCI_(345 or 6)comp.json`.

## Run

The finetuning code is located in directory `experiment`. 

### For LaMAGIC paper:

#### Train on 3, 4, 5-component circuits
1. 
```bash 
python experiment/lamagic1/trn_LLM_instruction.py
``` 
This is for Naïve formulation (NF), Canonical formulation (CF), and Canonical formulation + new duty cycle representation (CFDC)

2. 
```bash 
python experiment/lamagic1/trn_LLM_pure_text_matrix_form.py
``` 
This is for pure-text adjacency-matrix-based formulation (PM).

3. 
```bash
python experiment/lamagic1/trn_LLM_float_input_matrix_form.py
```
This is for float-input adjacency-matrix-based formulation (FM). This is the core contribution in LaMAGIC.

#### Then finetune on 6-component circuit with only 500, 1000, 2000 data points
1.
```bash
python experiment/lamagic1/trn_LLM_6_comp.py
```
This is for finetuning 6-component circuit on CF, PM, and FM.

### For LaMAGIC2 paper:

#### Train on 3, 4, 5-component circuits
```bash
python experiment/lamagic2/trn_pure_tranformer.py
```
This contains Succinct float-input adjacency-matrix-based formulation (SFM) and Succinct float-input canonical formulation with identifier (SFCI). SFCI is the core contribution of LaMAGIC2.

#### Finetune on 6-component circuits with only 500, 1000, and 2000 data points
```bash
python experiment/lamagic2/trn_pure_tranformer_6comp.py
```
This also contains SFM and SFCI.