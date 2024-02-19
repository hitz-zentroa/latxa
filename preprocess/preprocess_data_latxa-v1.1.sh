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

# Loop over each subfolder
for dir in ${WORK}/data/ilenia/v1/jsonl/*; do
    if [ -d "$dir" ]; then
        # Get the name of the subfolder
        subfolder=${dir##*/}

        # if bne_v1, skip
        if [ "$subfolder" == "bne_v1" ]; then
            continue
        fi

        echo "Processing ${subfolder}"

        # Create the output directory
        mkdir -p "$WORK/preprocessed_data/ilenia/v1/${subfolder}"

        # Define the data types
        splits=("train" "test" "valid")

        # if subfolder equals CONCAT add _eu to splits
        if [ "$subfolder" == "CONCAT" ]; then
            splits=("train_eu" "test_eu" "valid_eu")
        fi

        # Loop over the data types
        for split in "${splits[@]}"; do
            echo "Processing ${split}"

            # Calculate the number of lines in the file
            num_lines=$(zcat "$dir/${split}.jsonl.gz" | wc -l)

            # Preprocess the data
            python tools/preprocess_data.py \
                --input "$WORK/data/ilenia/v1/jsonl/${subfolder}/${split}.jsonl.gz" \
                --output-prefix "$WORK/preprocessed_data/ilenia/v1/${subfolder}/${split}" \
                --tokenizer-type "SPMTokenizer" \
                --vocab-file "/leonardo_scratch/large/userexternal/jetxaniz/Llama-2-7b/tokenizer.model" \
                --num-docs $num_lines \
                --append-eod \
                --workers 8
        done
    fi
done
