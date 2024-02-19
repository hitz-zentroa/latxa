path=/leonardo_scratch/large/userexternal/jetxaniz
input=Llama-2-70b-neox-eus-v0/global_step
output=Llama-2-70b-hf-eus-v0/global_step

source ${WORK}/environments/neox-env/bin/activate

# merge yml configs in ${path}/${input}${i}/configs into ${path}/${input}${i}/config.yml
python merge_configs.py \
    --input_dir ${path}/${input}100/configs \
    --output_dir ${path}/config.yml

cd ${WORK}/gpt-neox/tools

# loop global step from 100 to 1500 with step 100
for i in {100..2000..100}; do
    echo "Converting ${path}/${input}${i} to ${path}/${output}${i}"
    python convert_llama_gqa_sequential_to_hf.py \
        --input_dir ${path}/${input}${i} \
        --config_file ${path}/config.yml \
        --output_dir ${path}/${output}${i}
done
