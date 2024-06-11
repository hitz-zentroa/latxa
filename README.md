<p align="center">
    <br>
    <img src="assets/latxa_round.png" style="height: 350px;">
    <br>
    <h1 align="center">Latxa: An Open Language Model and Evaluation Suite for Basque</h1>


<p align="center">
    <a href="https://github.com/hitz-zentroa/latxa/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/latxa"></a>
    <a href="https://huggingface.co/collections/HiTZ/latxa-65a697e6838b3acc53677304"><img alt="Pretrained Models" src="https://img.shields.io/badge/🤗HuggingFace-Pretrained Models-green"></a>
    <a href="https://www.hitz.eus/en/node/340"><img alt="Blog" src="https://img.shields.io/badge/📒-Blog Post-blue"></a>
    <a href="https://arxiv.org/abs/2403.20266"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
<br>
     <a href="http://www.hitz.eus/"><img src="https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet"></a>
    <a href="http://www.ixa.eus/?language=en"><img src="https://img.shields.io/badge/IXA-%20NLP%20Group-ff3333"></a>
    <br>
     <br>
</p>

<p align="justify">
We introduce <img src="assets/latxa_round.png" width="18"> Latxa, a family of large language models for Basque ranging from 7 to 70 billion parameters. Latxa is based on Llama 2, which we continue pretraining on a new Basque corpus comprising 4.3M documents and 4.2B tokens. Addressing the scarcity of high-quality benchmarks for Basque, we further introduce 4 multiple choice evaluation datasets: EusProficiency, comprising 5,169 questions from official language proficiency exams; EusReading, comprising 352 reading comprehension questions; EusTrivia, comprising 1,715 trivia questions from 5 knowledge areas; and EusExams, comprising 16,046 questions from public examinations. In our extensive evaluation, Latxa outperforms all previous open models we compare to by a large margin. In addition, it is competitive with GPT-4 Turbo in language proficiency and understanding, despite lagging behind in reading comprehension and knowledge-intensive tasks. Both the Latxa family of models, as well as our new pretraining corpora and evaluation datasets, are publicly available under open licenses. Our suite enables reproducible research on methods to build LLMs for low-resource languages.

- 📒 Blog Post: [Latxa: An Open Language Model and Evaluation Suite for Basque](https://www.hitz.eus/en/node/343)
- 📖 Paper: [Latxa: An Open Language Model and Evaluation Suite for Basque](https://arxiv.org/abs/2403.20266)
- <img src="assets/latxa_round.png" width="15"> Latxa in the 🤗HuggingFace Hub: [HiTZ/latxa](https://huggingface.co/collections/HiTZ/latxa-65a697e6838b3acc53677304)
</p>

# Getting started

Use the code below to get started with the model.
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="HiTZ/latxa-7b-v1.1")

text = "Euskara adimen artifizialera iritsi da!"

pipe(text, max_new_tokens=50, num_beams=5)

>> [
 {
  'generated_text': 'Euskara adimen artifizialera iritsi da!\nEuskararen eta adimen artifizialaren arteko harremana aspaldikoa da,'
  ' baina azken urteotan aurrerapauso handiak eman dira arlo horretan'
 }
]
```

# Training

Code for training models on the [CINECA HPC Leonardo](https://wiki.u-gov.it/confluence/display/SCAIUS/UG3.2%3A+LEONARDO+UserGuide) cluster using [GPT-Neox](https://github.com/EleutherAI/gpt-neox). If you train on another cluster you will need to update some settings. Check the GPT-Neox documentation if you have any doubts. 

The training process is divided into several steps: loading the required modules, creating a virtual environment, installing GPT-Neox, downloading the Llama models, converting the checkpoints, downloading the data, preprocessing the data, defining the training configs, setting up wandb, running the training, and converting the Neox checkpoints to HF.

## Load Modules

Install modules needed for GPT-Neox. You can add this to `.bashrc` so that modules are loaded automatically:

```bash
module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs
```

## Create Virtual Environment

Create a virtual environment:

```bash
python -m venv $WORK/environments/neox-env
```

Activate the virtual environment. You can add this to `.bashrc` so that the virtual environment is activated automatically:

```bash
source $WORK/environments/neox-env/bin/activate
```

## Install GPT-Neox

Clone the repository and install the requirements:

```bash
git clone https://github.com/EleutherAI/gpt-neox
cd gpt-neox
```

Install the requirements:

```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-wandb.txt
pip install -r requirements/requirements-flashattention.txt
python ./megatron/fused_kernels/setup.py install # optional, if using fused kernels
```

## Setup Wandb

As the Leonardo cluster does not have internet connection, offline mode must be used. Change default wandb config directory:

```bash
export WANDB_CONFIG_DIR=$HOME/.wandb
```

Login to wandb:

```bash
wandb login
```

Change wandb to offline mode:

```bash
wandb offline
```

Logs will be saved to `$WORK/gpt-neox/wandb` directory.

To sync logs to wandb, run:

```bash
wandb sync $WORK/gpt-neox/wandb
```

## Download Llama Models

Download the raw Llama models from the following links:

Llama-2-7b: https://huggingface.co/meta-llama/Llama-2-7b

Llama-2-13b: https://huggingface.co/meta-llama/Llama-2-13b

Llama-2-70b: https://huggingface.co/meta-llama/Llama-2-70b

## Convert Checkpoints

Convert the raw Llama models to Neox format using scripts in `convert` directory.

```bash
cd convert
bash convert_raw_llama_weights_to_neox_7b.sh
bash convert_raw_llama_weights_to_neox_13b.sh
bash convert_raw_llama_weights_to_neox_70b.sh
```

## Download Data

Download the pretraining data from the following links:

Euscrawl: https://huggingface.co/datasets/HiTZ/euscrawl

Pile: https://huggingface.co/datasets/EleutherAI/pile

Latxa v1.1: https://huggingface.co/datasets/HiTZ/latxa-corpus-v1.1

## Preprocess Data

Preprocess data using scripts available in `preprocess` directory:

To preprocess the Pile dataset, run `bash preprocess_data_pile.sh`.

To preprocess the Latxa v1 dataset, run `bash preprocess_data_latxa-v1.sh`.

To preprocess the Latxa v1.1 dataset, run `bash preprocess_data_latxa-v1.1.sh`.

## Define Configs

Define training configs in the `configs` directory. You can use the existing configs as a template. There are two base configs which are common to all models, and include details such as checkpointing and logging. The first one is used for 7B and 13B models, and the other one for 70B models. Additional configs are divided into 4 folders, depending on the type of parameters: data, deepspeed, hyperparameters and models.

- `data`: contains the data configuration files for Latxa v1 and v1.1.
- `deepspeed`: contains the deepspeed configuration file for Zero 1.
- `hyperparameters`: contains the hyperparameters configuration files for Latxa v1 and v1.1 of three sizes.
- `models`: contains the model configuration files for three sizes.

## Run Training

Run training using scripts available in `train` directory. There are scripts for Latxa v1 and v1.1 models of three sizes. For example, to train Latxa 7B v1.1, run:

```bash
cd train/latxa-7b
bash llama-2-7b-v1.1.sh
```

## Convert Neox Checkpoints to HF

The Neox checkpoints can be converted to HF using the `convert_neox_to_hf.py` script. The script take an input path, the output path, the model config, the precision and the architecture as arguments. You can find example scripts in the `convert` directory. For example, to convert the Latxa 7B v1.1 model, run:

```bash
cd convert
bash convert_neox_to_hf_7b_v1.1.sh
```

# Evaluation

Evaluation scripts for open models are in the `scripts` directory. `openai` directory contains scripts for evaluating openai models. Evaluation results are in the `results` directory.

## Install Evaluation Harness

You will need to install [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Clone the repository and install the requirements:

```bash	
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

## Run Open Model Evaluation

To run evaluation on open models, use the scripts in the `scripts` directory. Each script evaluates a model in all the tasks. For example, to run evaluation on Latxa v1.1 7b, run:

```bash
sbatch lm_eval_latxa-7b-v1.1.slurm
```

## Check Evaluation Results

Evaluation results are in the `results` directory. Each model has a directory with the results of the evaluation in each task. The results are in the form of a json file with the average scores of the model in each task.

## Run OpenAI Model Evaluation

To run evaluation on OpenAI models, use the scripts in the `openai` directory. There is a python script to evaluate each dataset, and a bash script for each model and dataset. For example, to run evaluation on GPT-3.5 Turbo on EusTrivia, run:

```bash
bash gpt-3.5-turbo-0125_eus_trivia.sh
```

## Check OpenAI Evaluation Results

Evaluation results are in the `results` directory. Each model has a directory with the results of the evaluation in each task. In this case, all the outputs of the models are saved for each task. Scores can be calculated using the `correct` field. For EusTrivia and EusExams, there are additional scripts to obtained detailed results by category. For example, to get detailed results for GPT-3.5 Turbo on EusTrivia, run:

```bash
python calculate_accuracy_eus_trivia.py
```

# Versioning
We keep updating and improving our base model, this section covers the major releases and changes we made. We recommend users always to use the latest version available.

* [v1.2-latest](https://huggingface.co/HiTZ/latxa-7b-v1.2): (bug-fix in v1.1) Avoids generating JSON-like responses.
* [v1.1](https://huggingface.co/HiTZ/latxa-7b-v1.1): (improvement) The model is trained with much more corpora. Please take a look at the ACL paper for more details.
* [v1.0](https://huggingface.co/HiTZ/latxa-7b-v1): Initial version.

# Acknowledgements

This work has been partially supported by the Basque Government (IKER-GAITU project). It has also been partially supported by the Ministerio para la Transformación Digital y de la Función Pública - Funded by EU – NextGenerationEU within the framework of the project with reference 2022/TL22/00215335. The models were trained on the Leonardo supercomputer at CINECA under the EuroHPC Joint Undertaking, project EHPC-EXT-2023E01-013.

# Citation

To cite our work, please use:

```bibtex
@misc{etxaniz2024latxa,
      title={Latxa: An Open Language Model and Evaluation Suite for Basque}, 
      author={Julen Etxaniz and Oscar Sainz and Naiara Perez and Itziar Aldabe and German Rigau and Eneko Agirre and Aitor Ormazabal and Mikel Artetxe and Aitor Soroa},
      year={2024},
      eprint={2403.20266},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
