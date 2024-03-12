#!/bin/bash
#SBATCH -A EUHPC_E02_013
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00         # format: HH:MM:SS
#SBATCH --nodes 1               # 4 node
#SBATCH --ntasks-per-node=4     # 4 tasks out of 32
#SBATCH --gres=gpu:4            # 4 gpus per node out of 4
#SBATCH --mem=494000MB              # memory per node out of 494000MB
#SBATCH --output=.slurm/preprocess_data_pile.out
#SBATCH --error=.slurm/preprocess_data_pile.err
#SBATCH --exclusive
#SBATCH --requeue

# load leonardo modules
module load profile/deeplrn
module load python/3.10.8--gcc--11.3.0
module load cuda/11.8
module load openmpi/4.1.4--gcc--11.3.0-cuda-11.8
module load zlib/1.2.13--gcc--11.3.0
module load git-lfs

# Move to the gpt-neox install  
cd ${WORK}/gpt-neox

# setup the virtual env
source ${WORK}/environments/neox-env/bin/activate

# train data sizes: 100, 500
sizes=(100 500)

for size in "${sizes[@]}"; do
    echo "Preprocessing thepile size $size"
    python tools/datasets/preprocess_data.py \
        --input "$WORK/data/thepile/train-sample-${size}k.jsonl" \
        --output-prefix "$WORK/preprocessed_data/thepile/train-sample-${size}k" \
        --tokenizer-type "SPMTokenizer" \
        --vocab-file "/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b/tokenizer.model" \
        --num-docs ${size}000 \
        --append-eod \
        --workers 8
done
