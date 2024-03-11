path=/leonardo_scratch/large/userexternal/jetxaniz/
cd ${WORK}/gpt-neox/

source ${WORK}/environments/neox-env/bin/activate

cd tools/ckpts/

for TP in 1 2 4; do
    python convert_raw_llama_weights_to_neox.py \
        --input_dir ${path}Llama-2-13b \
        --model_size 13B \
        --output_dir ${path}Llama-2-13b-neox-TP-${TP} \
        --num_output_shards ${TP}
done
