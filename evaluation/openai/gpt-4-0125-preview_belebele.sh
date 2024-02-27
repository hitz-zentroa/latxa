export HF_DATASETS_CACHE="/gaueko0/users/jetxaniz007/.cache/huggingface/datasets/"

python evaluate_belebele.py \
    --split eus_Latn \
    --model gpt-4-0125-preview \
    --shots 5 \
    --limit 0 \
    --start 338
