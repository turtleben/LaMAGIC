# LaMAGIC and LaMAGIC2: Language-Model-based Topology Generation for Analog Integrated Circuits

### Published at ICML 2024 & ICML 2025

This repository provides the official implementation of LaMAGIC and LaMAGIC2, two approaches leveraging Language Models (LMs) for automated topology generation of analog integrated circuits.

* **LaMAGIC** ([ICML'24 Paper](https://arxiv.org/pdf/2407.18269))
* **LaMAGIC2** ([ICML'25 Paper](https://arxiv.org/abs/2506.10235))

---

## Installation

### Environment Setup

First, create a Conda environment using:

```bash
conda env create -f environment.yml
```

---

## Dataset and Model Setup

### Base Model

Download the pretrained `flan-T5-base` model from [Hugging Face](https://huggingface.co/google/flan-t5-base). Update the model path in all YAML configuration files located in:

```
analog_LLM/configs
```

### Dataset

Clone the dataset from [LaMAGIC-dataset](https://huggingface.co/datasets/turtleben/LaMAGIC-dataset).

Target data is located at:

```
[your_save_path]/LaMAGIC-dataset/transformed
```

* For LaMAGIC2 SOTA experiments, use:

  * `LaMAGIC2/SFCI_345comp.json`
  * `LaMAGIC2/SFCI_6comp.json`

---

## Training

Training scripts are organized by each paper under the `experiment` directory.

### LaMAGIC (ICML'24)

#### Initial training on 3, 4, and 5-component circuits:

* **Naïve (NF), Canonical (CF), Canonical + Duty Cycle (CFDC)**:

```bash
python experiment/lamagic1/trn_LLM_instruction.py
```

* **Pure-text adjacency-matrix (PM)**:

```bash
python experiment/lamagic1/trn_LLM_pure_text_matrix_form.py
```

* **Float-input adjacency-matrix (FM)** *(core contribution)*:

```bash
python experiment/lamagic1/trn_LLM_float_input_matrix_form.py
```

#### Fine-tuning on 6-component circuits (limited data: 500, 1000, 2000 samples):

```bash
python experiment/lamagic1/trn_LLM_6_comp.py
```

---

### LaMAGIC2 (ICML'25)

#### Initial training on 3, 4, and 5-component circuits:

* **Succinct Float-input Matrix (SFM), Succinct Float-input Canonical with Identifier (SFCI)** *(core contribution)*:

```bash
python experiment/lamagic2/trn_pure_tranformer.py
```

#### Fine-tuning on 6-component circuits (limited data: 500, 1000, 2000 samples):

```bash
python experiment/lamagic2/trn_pure_tranformer_6comp.py
```

#### Released Model Checkpoints:

* **3, 4, 5-component circuits (SFM):**

  1. Trained on SFM with data augmentation (vertex random shuffle): [LaMAGIC2-345comp-SFM-dataaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFM-dataaug)
  2. After data augment, trained on SFM without data augmentation: [LaMAGIC2-345comp-SFM-noaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFM-noaug)

* **3, 4, 5-component circuits (SFCI):**

  1. Trained on SFCI with data augmentation (vertex random shuffle): [LaMAGIC2-345comp-SFCI-dataaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFCI-dataaug)
  2. After data augment, trained on SFCI without data augmentation: [LaMAGIC2-345comp-SFCI-noaug](https://huggingface.co/turtleben/LaMAGIC2-345comp-SFCI-noaug)

* **6-component circuits (SFM):**

  * [LaMAGIC2-6Comp-SFM-dnum500](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFM-dnum500)
  * [LaMAGIC2-6Comp-SFM-dnum1000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFM-dnum1000)
  * [LaMAGIC2-6Comp-SFM-dnum2000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFM-dnum2000)

* **6-component circuits (SFCI):**

  * [LaMAGIC2-6Comp-SFCI-dnum500](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFCI-dnum500)
  * [LaMAGIC2-6Comp-SFCI-dnum1000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFCI-dnum1000)
  * [LaMAGIC2-6Comp-SFCI-dnum2000](https://huggingface.co/turtleben/LaMAGIC2-6Comp-SFCI-dnum2000)


---

## Citation

If you use this work in your research, please cite our papers:

```bibtex
@InProceedings{chang2024lamagic,
  title = 	 {{L}a{MAGIC}: Language-Model-based Topology Generation for Analog Integrated Circuits},
  author =       {Chang, Chen-Chia and Shen, Yikang and Fan, Shaoze and Li, Jing and Zhang, Shun and Cao, Ningyuan and Chen, Yiran and Zhang, Xin},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {6253--6262},
  year = 	 {2024},
  month = 	 {21--27 Jul},
  organization =    {PMLR},
}

@inproceedings{chang2025lamagic2,
  title={{L}a{MAGIC}2: Advanced Circuit Formulations for Language Model-Based Analog Topology Generation},
  author={Chang, Chen-Chia and Lin, Wan-Hsuan and Shen, Yikang and Chen, Yiran and Zhang, Xin},
  booktitle={Proceedings of the 42st International Conference on Machine Learning},
  year={2025},
  organization =    {PMLR},
}
```

---

## Contact

For questions or further collaboration, please reach out to the authors.
