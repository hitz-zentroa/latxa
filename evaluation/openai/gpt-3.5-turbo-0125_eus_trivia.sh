export HF_DATASETS_CACHE="/gaueko0/users/jetxaniz007/.cache/huggingface/datasets/"

python evaluate_eus_trivia.py \
    --split test \
    --model gpt-3.5-turbo-0125 \
    --shots 5 \
    --limit 0
