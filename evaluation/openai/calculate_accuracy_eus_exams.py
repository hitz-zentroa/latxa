import os
import json

def calculate_accuracy(model_name):
    with open(f'eus_exams_{model_name}.txt', 'w') as output_file:
        for filename in sorted(os.listdir(f'../results/{model_name}')):
            if filename.startswith('eus_exams') and filename.endswith('.jsonl'):
                correct_count = 0
                total_count = 0
                with open(os.path.join(f'../results/{model_name}', filename), 'r') as file:
                    for line in file:
                        data = json.loads(line)
                        if 'correct' in data:
                            total_count += 1
                            if data['correct']:
                                correct_count += 1

                if total_count > 0:
                    accuracy = correct_count / total_count
                    output_file.write(f'{round(accuracy * 100, 2)}\n')
                    print(f'Accuracy for {filename}: {round(accuracy * 100, 2)}')
                else:
                    output_file.write('No data found\n')

calculate_accuracy('gpt-3.5-turbo-0125')
calculate_accuracy('gpt-4-0125-preview')