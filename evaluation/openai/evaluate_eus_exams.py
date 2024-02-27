from openai import OpenAI
import os
from datasets import load_dataset
import random
import argparse
import json
import datetime
import datasets
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

seed = 42
random.seed(seed)

# Set your OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

answer2letter = {0: "A", 1: "B", 2: "C", 3: "D"}
letters = ["A", "B", "C", "D"]

print("Loading the configs from json file...")
# load the configs from the json file
with open("eus_exams_configs_eu.json", "r") as f:
    configs = json.load(f)

CONFIGS = configs["configs"]

print(CONFIGS)

def load_eus_exams():
    # Load your dataset configs from Hugging Face into a dict of datasets.Dataset
    dataset = {}
    for config in CONFIGS:
        dataset[config] = load_dataset("HiTZ/EusExams", name=config, split="test")
        dataset[config] = process_docs(dataset[config])
    return dataset

def process_docs(dataset: datasets.Dataset):
    """Filter out examples with no answer."""

    def valid_example(example: dict) -> bool:
        """Check if an example is valid."""
        if example["answer"] not in [0, 1, 2, 3]:
            return False
        if example["candidates"] == ["", "", "", ""]:
            return False
        return True

    return dataset.filter(valid_example)


def format_question(item):
    question = item["question"]
    candidates = item["candidates"]

    formatted_question = f"Galdera: {question}\nA: {candidates[0]}\nB: {candidates[1]}\nC: {candidates[2]}\nD: {candidates[3]}\nErantzuna:"
    return formatted_question


def few_shot_messages(few_shot_examples):
    # Add the few-shot examples to the prompt
    messages = [
        {
            "role": "system",
            "content": "Respond always with a single letter: A, B, C or D.",
        }
    ]
    for example in few_shot_examples:
        messages.append({"role": "user", "content": format_question(example)})
        messages.append(
            {"role": "assistant", "content": answer2letter[example["answer"]]}
        )
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def datetime_handler(x):
    if isinstance(x, datetime.datetime):
        return x.date().isoformat()
    raise TypeError("Unknown type")


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


def evaluate_eus_exams(split="test", model="gpt-3.5-turbo", shots=5, limit=1, start=0):
    # Load your dataset from Hugging Face
    datasets = load_eus_exams()
    
    tokens = 0
    cost = 0    

    # Create the results directory if it doesn't exist
    os.makedirs(f"../results/{model}", exist_ok=True)

    for config, dataset in datasets.items():
        print(f"Evaluating {config} split...")
        # Iterate over your dataset and use the API
        for i, item in enumerate(dataset):
            if i < start:
                continue

            # Get 5 random few-shot examples
            few_shot_examples = random.sample([ex for ex in dataset if ex != item], shots)

            # Add the few-shot examples to the prompt
            messages = few_shot_messages(few_shot_examples)

            messages.append({"role": "user", "content": format_question(item)})

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
            item["correct"] = (
                response["choices"][0]["message"]["content"]
                == answer2letter[item["answer"]]
            )
            
            # Calculate OpenAI API cost and add to the item
            item["cost"] = openai_api_calculate_cost(completion.usage, model)
            
            cost += item["cost"]
            tokens += completion.usage.total_tokens
            
            # Print details in a line: i, total tokens and total cost
            print(f"{i + 1}: ${cost:.4f} total cost, {tokens:,} tokens")
            
            with open(f"../results/{model}/eus_exams_{config}_{split}_{shots}-shot.jsonl", "a") as f:
                json.dump(item, f, default=datetime_handler)
                f.write("\n")

            if i == limit - 1:
                break


def main():
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
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
    parser.add_argument(
        "--start", type=int, default=0, help="Start index of the examples to evaluate"
    )
    args = parser.parse_args()
    evaluate_eus_exams(
        split=args.split, model=args.model, shots=args.shots, limit=args.limit, start=args.start
    )


if __name__ == "__main__":
    main()