import os
import json
from collections import defaultdict

def calculate_accuracy(model_name):
    category_results = defaultdict(lambda: {'correct_count': 0, 'total_count': 0})
    difficulty_results = defaultdict(lambda: {'correct_count': 0, 'total_count': 0})

    for filename in sorted(os.listdir(f'../results/{model_name}')):
        if filename.startswith('eus_trivia') and filename.endswith('.jsonl'):
            with open(os.path.join(f'../results/{model_name}', filename), 'r') as file:
                for line in file:
                    data = json.loads(line)
                    if 'correct' in data:
                        category_results[data['category']]['total_count'] += 1
                        difficulty_results[data['difficulty']]['total_count'] += 1
                        if data['correct']:
                            category_results[data['category']]['correct_count'] += 1
                            difficulty_results[data['difficulty']]['correct_count'] += 1

    with open(f'eus_trivia_{model_name}_accuracy.txt', 'w') as output_file:
        for category, counts in category_results.items():
            if counts['total_count'] > 0:
                accuracy = counts['correct_count'] / counts['total_count']
                output_file.write(f'Accuracy for category {category}: {round(accuracy * 100, 2)}\n')
                print(f'Accuracy for category {category}: {round(accuracy * 100, 2)}')

        for difficulty, counts in difficulty_results.items():
            if counts['total_count'] > 0:
                accuracy = counts['correct_count'] / counts['total_count']
                output_file.write(f'Accuracy for difficulty {difficulty}: {round(accuracy * 100, 2)}\n')
                print(f'Accuracy for difficulty {difficulty}: {round(accuracy * 100, 2)}')

calculate_accuracy('gpt-3.5-turbo-0125')
calculate_accuracy('gpt-4-0125-preview')