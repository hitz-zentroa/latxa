import html
import re

from openai import OpenAI
import os
from datasets import load_dataset, load_metric
import random
import argparse
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tqdm import tqdm

seed = 42
random.seed(seed)

# Set your OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def openai_api_calculate_cost(usage, model="gpt-4-0125-preview"):
    pricing = {
        'gpt-3.5-turbo-0125': {
            'prompt': 0.0005,
            'completion': 0.0015,
        },
        'gpt-4-0125-preview': {
            'prompt': 0.01,
            'completion': 0.03,
        },
        'gpt-4-0613': {
            'prompt': 0.03,
            'completion': 0.06,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")

    prompt_cost = usage.prompt_tokens * model_pricing['prompt'] / 1000
    completion_cost = usage.completion_tokens * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    # round to 6 decimals
    total_cost = round(total_cost, 6)

    return total_cost


def micro_f1_score(items):
    f1_metric = load_metric("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, average="micro")["f1"]
    return f1_score


def vaxx_f1_score(items):
    f1_metric = load_metric("f1")
    golds, preds = list(zip(*items))
    f1_score = f1_metric.compute(references=golds, predictions=preds, labels=[0, 2], average="macro")["f1"]
    return f1_score


def accuracy_score(items):
    arr = [int(g == p) for g, p in items]
    return sum(arr) / len(arr)


def general_detokenize(string):
    string = re.sub(r'\s+([.,;:!?)])', r'\1', string)
    string = re.sub(r'(\s+|^)\(\s+([^)]+)\s+\)', r'\1(\2)', string)
    string = re.sub(r'(\s+|^)\[\s+([^)]+)\s+\]', r'\1[\2]', string)
    string = re.sub(r'(\s+|^)"\s+([^"]+)\s+"', r'\1"\2"', string)
    string = re.sub(r"(\s+|^)'\s+([^']+)\s+'", r"\1'\2'", string)
    return string


def process_doc(string):
    string = html.unescape(string)
    string = general_detokenize(string)
    return string


def process_wic_docs(dataset):
    def _helper(doc):
        # there's some issues with the encoding on this one
        doc["sentence1"] = process_doc(doc["sentence1"]).encode('latin-1').decode('utf-8')
        doc["sentence2"] = process_doc(doc["sentence2"]).encode('latin-1').decode('utf-8')
        return doc
    return dataset.map(_helper)


def load_basqueglue(name):
    return load_dataset("orai-nlp/basqueGLUE", name=name, split="test")


def format_question_bec(item):
    return f"Testua: {item['text']}\nGaldera: Nolako jarrera agertzen du aurreko testuak?\nErantzuna:"


def format_question_bhtc(item):
    labels = ', '.join(CONFIGS['bhtc'][2])
    return f"Testua: {item['text']}\nGaldera: Zein da aurreko testuaren gaia? Aukeratu hauen artean: {labels}\nErantzuna:"


def format_question_coref(item):
    def _span_in_context(span_index, span_text):
        span_start = span_index
        span_end = span_start + len(span_text.split(" ")) - 1
        tokens[span_start] = f'*{tokens[span_start]}'
        tokens[span_end] = f'{tokens[span_end]}*'
    tokens = item["text"].split(" ")
    _span_in_context(item["span1_index"], item["span1_text"])
    _span_in_context(item["span2_index"] - 1, item["span2_text"])  # span1_index is 0-based but span2_index is 1-based ??
    context = process_doc(" ".join(tokens))
    span_1 = process_doc(item["span1_text"])
    span_2 = process_doc(item["span2_text"])
    text = (
        f'Testua: {context}\nGaldera: Aurreko testuan, "*{span_1}*" eta "*{span_2}*" gauza bera dira?\nErantzuna:'
    )
    return text


def format_question_qnli(item):
    return f"{item['question']}\n{item['sentence']}\nGaldera: aurreko galderari erantzuten al dio emandako testuak?\nErantzuna:"


def format_question_vaxx(item):
    return f"Testua: {item['text']}\nGaldera: Nolako jarrera agertzen du aurreko testuak txertoei buruz?\nErantzuna:"


def format_question_wic(item):
    return f"1. esaldia: {item['sentence1']}\n2. esaldia: {item['sentence2']}\nGaldera: Aurreko bi esaldietan, \"{item['word']}\" hitzak esanahi berdina du?\nErantzuna:"


def few_shot_messages(few_shot_examples, format_question_func, labels):
    labels_str = ', '.join(labels)
    messages = [{"role": "system", "content": f"Respond always with one of these: {labels_str}"}]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": format_question_func(example)})
        answer = labels[example['label']] if isinstance(example['label'], int) else example['label']
        messages.append({"role": "assistant", "content": answer})
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def evaluate_basqueglue(split, model="gpt-3.5-turbo", shots=5, limit=1):
    # Load your dataset from Hugging Face
    print(f"Loading {split} split...")
    dataset = load_basqueglue(name=split)
    if split == 'wic':
        dataset = process_wic_docs(dataset)

    # Create the results directory if it doesn't exist
    os.makedirs(f"../results/{model}", exist_ok=True)

    format_question_func, eval_func, possible_answers = CONFIGS[split]
    y_gold_and_pred = []
    tokens = cost = 0

    # Iterate over your dataset and use the API
    for i, item in tqdm(enumerate(dataset)):

        # Get 5 random few-shot examples
        few_shot_examples = random.sample([ex for ex in dataset if ex != item], shots)

        # Add the few-shot examples to the prompt
        messages = few_shot_messages(few_shot_examples, format_question_func, possible_answers)

        messages.append({"role": "user", "content": format_question_func(item)})

        # Save messages along with the original dataset fields to a jsonl file
        item["messages"] = messages

        # Use the chat models, which are better for multi-turn conversation
        completion = completion_with_backoff(
            model=model,
            messages=messages,
            temperature=0,
            seed=seed,
        )

        # convert completions to dict
        response = completion.model_dump()

        # Save whole response along with the original dataset fields to a jsonl file
        item["response"] = response

        # Check if the answer is correct
        answer_idx = item["label"]
        answer = possible_answers[answer_idx]
        prediction = response["choices"][0]["message"]["content"]
        item["correct"] = prediction == answer

        try:
            pred_idx = possible_answers.index(prediction)
        except ValueError:
            # choose different to actual answer so that it counts as an error (very rarely occurs)
            print(f'"{prediction}" is not in {possible_answers}')
            pred_idx = int(not bool(answer_idx))
        y_gold_and_pred.append((answer_idx, pred_idx))

        # Calculate OpenAI API cost and add to the item
        item["cost"] = openai_api_calculate_cost(completion.usage, model)

        cost += item["cost"]
        tokens += completion.usage.total_tokens

        # Print details in a line: i, total tokens and total cost
        print(f"{i + 1}: ${cost:.4f} total cost, {tokens:,} tokens")

        with open(f"../results/{model}/basqueglue_{split}_{shots}-shot__TEST.jsonl", "a") as f:
            json.dump(item, f)
            f.write("\n")

        if i == limit - 1:
            break

    print(eval_func.__name__, eval_func(y_gold_and_pred))


CONFIGS = {
    "bec": (format_question_bec, micro_f1_score, ['negatiboa', 'neutrala', 'positiboa']),
    "bhtc": (format_question_bhtc, micro_f1_score, [
        'Ekonomia', 'Euskal Herria', 'Euskara', 'Gizartea', 'Historia', 'Ingurumena', 'Iritzia', 'Komunikazioa', 'Kultura', 'Nazioartea', 'Politika', 'Zientzia'
    ]),
    "coref": (format_question_coref, accuracy_score, ['ez', 'bai']),
    "qnli": (format_question_qnli, accuracy_score, ['bai', 'ez']),
    "vaxx": (format_question_vaxx, vaxx_f1_score, ['aurka', 'neutrala', 'alde']),
    "wic": (format_question_wic, accuracy_score, ['ez', 'bai'])
}


def main():
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, help="Dataset split to evaluate on", choices=CONFIGS.keys(), required=True
    )
    parser.add_argument(
        "--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use"
    )
    parser.add_argument(
        "--shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--limit", type=int, default=1, help="Number of examples to evaluate"
    )
    args = parser.parse_args()
    evaluate_basqueglue(
        split=args.split, model=args.model, shots=args.shots, limit=args.limit
    )


if __name__ == "__main__":
    main()
