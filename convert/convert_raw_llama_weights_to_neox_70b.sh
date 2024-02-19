#!/bin/bash
#SBATCH -A EUHPC_E02_013
#SBATCH -p boost_usr_prod
#SBATCH -N 1                    # 1 node
#SBATCH --time 24:00:00         # format: HH:MM:SS
#SBATCH --ntasks-per-node=4     # 4 tasks
#SBATCH --gres=gpu:4            # 4 gpus per node out of 4
#SBATCH --mem=494000MB          # memory per node out of 494000MB
#SBATCH --output=.slurm/convert_raw_llama_weights_to_neox_70b.out
#SBATCH --error=.slurm/convert_raw_llama_weights_to_neox_70b.err
#SBATCH --exclusive
#SBATCH --requeue

path=/leonardo_scratch/large/userexternal/jetxaniz/
cd ${WORK}/gpt-neox/

source ${WORK}/environments/neox-env/bin/activate

cd tools

for TP in 4; do
    python convert_raw_llama_gqa_weights_to_neox.py \
        --input_dir ${path}Llama-2-70b \
        --model_size 70B \
        --output_dir ${path}Llama-2-70b-neox-TP-${TP}-PP \
        --num_output_shards ${TP} \
        --pipeline_parallel
done
