import os
import json
import argparse
from datasets import load_dataset

MATH_INSTRUCTION = "Please solve this math problem step by step. Give your final answer as a number after '####'"

def parse_args():
    parser = argparse.ArgumentParser(description='Convert GSM8K dataset to JSONL format')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory to save the output JSONL file')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("Loading GSM8K dataset...")
    gsm8k_ds = load_dataset("openai/gsm8k", "main")
    print("Dataset loaded successfully")

    # Create output directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    output_file = os.path.join(args.data_dir, "raw.jsonl")

    # Combine train and test data
    all_data = []
    
    # Process training data
    print("Processing training data...")
    for item in gsm8k_ds["train"]:
        example = {
            "instruction": MATH_INSTRUCTION,
            "input": item["question"],
            "output": item["answer"],
        }
        all_data.append(example)
    
    # # Process test data
    # print("Processing test data...")
    # for item in gsm8k_ds["test"]:
    #     example = {
    #         "instruction": MATH_INSTRUCTION,
    #         "input": item["question"],
    #         "output": item["answer"],
    #     }
    #     all_data.append(example)

    # Write to JSONL file
    print(f"Writing {len(all_data)} examples to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in all_data:
            f.write(json.dumps(example) + '\n')

    print("Conversion complete!")
    print(f"Total examples: {len(all_data)}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main()

# Example usage:
# python convert_gsm8k_to_jsonl.py --data_dir /path/to/output/dir
# python tools/convert_gsm8k_to_jsonl.py --data_dir data/gsm8k/train
# To include test data:
# python convert_gsm8k_to_jsonl.py --data_dir /path/to/output/dir --include_test