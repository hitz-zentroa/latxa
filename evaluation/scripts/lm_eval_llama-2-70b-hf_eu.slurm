#!/bin/bash
#SBATCH --job-name=lm_eval_llama-2-70b-hf_eu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0
#SBATCH --mem=100GB
#SBATCH --gres=gpu:4
#SBATCH --output=.slurm_acl/lm_eval_llama-2-70b-hf_eu.out
#SBATCH --error=.slurm_acl/lm_eval_llama-2-70b-hf_eu.err

# activate virtual environment
source /gaueko0/users/jetxaniz007/phd/venv2/bin/activate
export TRANSFORMERS_CACHE="/gaueko1/transformers_cache/"
export HF_DATASETS_CACHE="/sc04a2/users/jibalari/BACKUP/ilenia/llama-eus/hf_datasets_cache"
export TOKENIZERS_PARALLELISM=false

path="meta-llama"
model="Llama-2-70b-hf"
model_name=$path/$model


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
