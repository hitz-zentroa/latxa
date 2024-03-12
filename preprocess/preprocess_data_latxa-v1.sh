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

# train, validation, and test data
for split in "train" "valid" "test"; do
    # Calculate the number of lines in the file
    num_lines=$(zcat "$dir/${split}.jsonl.gz" | wc -l)

    python tools/datasets/preprocess_data.py \
        --input "$WORK/data/euscrawl/euscrawl/${split}.jsonl" \
        --output-prefix "$WORK/preprocessed_data/euscrawl/${split}/" \
        --tokenizer-type "SPMTokenizer" \
        --vocab-file "/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b/tokenizer.model" \
        --num-docs $num_lines \
        --append-eod \
        --workers 8
done
