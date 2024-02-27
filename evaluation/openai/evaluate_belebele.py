from openai import OpenAI
import os
from datasets import load_dataset
import random
import argparse
import json
import datetime
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


def load_belebele(name="eus_Latn"):
    # Load your dataset from Hugging Face
    dataset = load_dataset("facebook/belebele", name="default", split=name)
    return dataset


def format_question(item):
    flores_passage = item["flores_passage"]
    question = item["question"]
    mc_answer1 = item["mc_answer1"]
    mc_answer2 = item["mc_answer2"]
    mc_answer3 = item["mc_answer3"]
    mc_answer4 = item["mc_answer4"]

    # Format the question with the given prompt
    formatted_question = f"P: {flores_passage}\nQ: {question.strip()}\nA: {mc_answer1}\nB: {mc_answer2}\nC: {mc_answer3}\nD: {mc_answer4}\nAnswer:"
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
        answer = ['1', '2', '3', '4'].index(example["correct_answer_num"])
        messages.append({"role": "user", "content": format_question(example)})
        messages.append(
            {"role": "assistant", "content": answer2letter[answer]}
        )
    return messages


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


def datetime_handler(x):
    if isinstance(x, datetime.datetime):
        return x.date().isoformat()
    raise TypeError("Unknown type")


def evaluate_belebele(split="eus_Latn", model="gpt-3.5-turbo", shots=5, limit=1, start=0):
    # Load your dataset from Hugging Face
    print(f"Loading {split} split...")
    dataset = load_belebele(name=split)

    # Create the results directory if it doesn't exist
    os.makedirs(f"../results/{model}", exist_ok=True)

    # Iterate over your dataset and use the API
    for i, item in enumerate(dataset):
        if i < start:
            continue
        print(f"Processing example {i}...")

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
        answer = ['1', '2', '3', '4'].index(item["correct_answer_num"])
        item["correct"] = (
            response["choices"][0]["message"]["content"]
            == answer2letter[answer]
        )

        with open(f"../results/{model}/belebele_{split}_{shots}-shot.jsonl", "a") as f:
            json.dump(item, f, default=datetime_handler)
            f.write("\n")

        if i == limit - 1:
            break


def main():
    # Define the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default="eus_Latn", help="Dataset split to evaluate on"
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
    evaluate_belebele(
        split=args.split, model=args.model, shots=args.shots, limit=args.limit, start=args.start
    )


if __name__ == "__main__":
    main()
