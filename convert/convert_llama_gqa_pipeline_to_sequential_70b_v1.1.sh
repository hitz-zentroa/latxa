#!/bin/bash
#SBATCH -A EUHPC_E02_013
#SBATCH -p boost_usr_prod
#SBATCH -N 1                    # 1 node
#SBATCH --time 24:00:00         # format: HH:MM:SS
#SBATCH --ntasks-per-node=4     # 4 tasks
#SBATCH --gres=gpu:4            # 4 gpus per node out of 4
#SBATCH --mem=494000MB          # memory per node out of 494000MB
#SBATCH --output=.slurm/convert_llama_gqa_pipeline_to_hf_70b.out
#SBATCH --error=.slurm/convert_llama_gqa_pipeline_to_hf_70b.err
#SBATCH --exclusive
#SBATCH --requeue

INPUT_DIR=Llama-2-70b-neox-eus-v1.1_500k/global_step10000
OUTPUT_DIR=${INPUT_DIR}_hf

source ${WORK}/environments/neox-env-2/bin/activate


echo "Converting pipeline to sequential"
python ${WORK}/gpt-neox/tools/convert_llama_gqa_pipeline_to_sequential.py \
    --input_dir ${SCRATCH}/${INPUT_DIR} \
    --output_dir ${SCRATCH}/${INPUT_DIR}_sequential \
    --num_shards 4

echo "Merging configs"
python ${HOME}/llama-eus/leonardo/convert/merge_configs.py \
    --input_dir ${SCRATCH}/${INPUT_DIR}/configs \
    --output_dir ${SCRATCH}/${INPUT_DIR}_sequential/config.yml

echo "Converting sequential to HuggingFace"
python ${WORK}/gpt-neox/tools/convert_llama_gqa_sequential_to_hf.py \
    --input_dir ${SCRATCH}/${INPUT_DIR}_sequential/ \
    --config_file ${SCRATCH}/${INPUT_DIR}_sequential/config.yml \
    --output_dir ${SCRATCH}/${OUTPUT_DIR} \
    --no_cuda
