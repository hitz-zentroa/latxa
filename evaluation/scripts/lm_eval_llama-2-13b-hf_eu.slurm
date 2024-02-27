#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2-13b-hf_eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=.slurm/lm_eval_llama-2-13b_eu.out
#SBATCH --error=.slurm/lm_eval_llama-2-13b_eu.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko1/transformers_cache/"
export HF_DATASETS_CACHE="/gaueko0/users/jetxaniz007/.cache/huggingface/datasets/"
export TOKENIZERS_PARALLELISM=false

path="meta-llama"
model="Llama-2-13b-hf"
model_name=$path/$model

# select tasks
tasks_selected=(
    "eus_proficiency"
    "eus_reading"
    "eus_trivia"
    "eus_exams_eu"
    "xstorycloze_eu"
    "belebele_eus_Latn"
    "bhtc_v2" 
    "bec2016eu" 
    "vaxx_stance" 
    "qnlieu" 
    "wiceu" 
    "epec_koref_bin"
)

for group_name in "${tasks_selected[@]}"; do
    if [[ $group_name == "xnli_eu" || $group_name == "xstorycloze_eu" ]]; then
        num_fewshot=0
    else
        num_fewshot=5
    fi
    srun python3 -m lm_eval \
        --model hf \
        --model_args pretrained=$model_name,parallelize=True \
        --tasks $group_name \
        --device cuda \
        --output_path ../results/llama-2/${model}/${model}_${group_name}_${num_fewshot}-shot.json \
        --batch_size auto \
        --num_fewshot ${num_fewshot} \
        --log_samples
done
