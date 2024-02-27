#!/bin/bash

source /home/nperez/.venv/openai/bin/activate

tasks=(
  "bec"
  "bhtc"
  "coref"
  "qnli"
  "vaxx"
  "wic"
)
model=gpt-3.5-turbo-0125
shots=5

for task in "${tasks[@]}"; do
  python evaluate_basqueglue.py --split "$task" --model $model --shots $shots --limit 0 > "../results/${model}/basqueglue_${task}_${shots}-shot.out"
done