import os
import yaml
import json

def merge_json_files(input_dir, output_dir):
    config_data = {}

    for filename in os.listdir(input_dir):
        if filename.endswith('.yml'):
            with open(os.path.join(input_dir, filename)) as f:
                data = yaml.safe_load(f)
                config_data.update(data)

    # dump in dict format
    with open(output_dir, 'w') as f:
        json.dump(config_data, f, indent=2)

# define main with arguments
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', help='input directory')
    parser.add_argument('--output_dir', '-o', help='output file')
    args = parser.parse_args()

    merge_json_files(args.input_dir, args.output_dir)
    
if __name__ == '__main__':
    main()