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

# validation data
python tools/preprocess_data.py \
    --input "$WORK/data/euscrawl/euscrawl-v1-free-jsonl/valid.jsonl" \
    --output-prefix "$WORK/preprocessed_data/euscrawl/valid/" \
    --tokenizer-type "SPMTokenizer" \
    --vocab-file "/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b/tokenizer.model" \
    --num-docs 10000 \
    --append-eod \
    --workers 8

# train data
python tools/preprocess_data.py \
    --input "$WORK/data/mixed_data/euscrawl_train_thepile_train100k.jsonl" \
    --output-prefix "$WORK/preprocessed_data/mixed_data/euscrawl_train_thepile_train100k/" \
    --tokenizer-type "SPMTokenizer" \
    --vocab-file "/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b/tokenizer.model" \
    --num-docs 1814545 \
    --append-eod \
    --workers 8