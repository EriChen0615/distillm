"""
Copied and modified from process_data_dolly.py
"""
# Debug mode toggle
DEBUG_MODE = False

import multiprocessing
import os
import time
import torch
import json
import sys
from numerize.numerize import numerize
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer
from arguments import get_args
from datasets import load_dataset

MATH_INSTRUCTION = "Please solve this math problem step by step. Give your final answer as a number."
# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object): 
    def __init__(self, args):
        self.args = args

    def initializer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

    def encode(self, line):
        if self.args.model_type!="qwen":
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )
        else:
            template = (
                "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Question:\n{input}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
            )
        prompt = template.format(
            instruction=MATH_INSTRUCTION,
            input=line['question']
        )
            
        response = line["answer"]
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Verify token IDs are valid
        if max(prompt_tokens) >= len(self.tokenizer):
            print(f"Max token ID in prompt: {max(prompt_tokens)}")
            print(f"Vocab size: {len(self.tokenizer)}")
            raise ValueError(f"Token ID {max(prompt_tokens)} exceeds vocab size {len(self.tokenizer)}")
        
        full_tokens = self.tokenizer.encode(prompt + response, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]
            # return None, None, None, None, len(line)
        
        if self.args.model_type == "qwen":
            separator_token = 160000  # or another suitable value that won't conflict with real token IDs
        else:
            separator_token = 65535
        
        return line, prompt, prompt_tokens, response_tokens, len(line)


def main():
    print("OK")
    args = get_args()
        
    if 'generated' not in args.processed_data_dir:
        args.processed_data_dir = os.path.join(args.processed_data_dir, args.model_type)

    os.makedirs(args.processed_data_dir, exist_ok=True)
    
    print("Loading dataset from:", args.data_dir)
    gsm8k_ds = load_dataset(args.data_dir, "main")
    print("Dataset loaded successfully")

    if args.dev_num > 0:
        all_data = {
            "valid": gsm8k_ds["test"].select(range(args.dev_num)),
            "train": gsm8k_ds["train"]
        }
    else:
        all_data = {
            "train": gsm8k_ds["train"]
        }
    
    for split in all_data:
        print(f"\nProcessing {split} split...")
        encoder = Encoder(args)
        encoder.initializer()

        proc_start = time.time()
        total_bytes_processed = 0
        
        bin_file = os.path.join(args.processed_data_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(args.processed_data_dir, f"{split}_{0}.idx")

        if args.model_type!="qwen":
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)
            separator_token = 65535  # max uint16
        else:
            binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint32)
            separator_token = 160000 # Qwen's vocab size (safe separator value)
            
        print(f"Using {binary_builder._dtype} builder")
        
        inst_num = 0  # Track actual processed examples
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        # First, collect all valid examples
        valid_examples = []
        for example in all_data[split]:
            line, prompt_str, prompt, response, bytes_processed = encoder.encode(example)
            if prompt is None:  # Skip invalid examples
                continue
                
            if args.only_prompt and len(prompt) >= args.max_length:
                continue
                
            valid_examples.append((line, prompt_str, prompt, response, bytes_processed))
        
        print(f"Found {len(valid_examples)} valid examples out of {len(all_data[split])} total")
        
        # Then process valid examples
        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(valid_examples):
            total_bytes_processed += bytes_processed
            
            if args.only_prompt:
                binary_builder.add_item(torch.IntTensor(prompt))
            else:
                if args.model_type == "qwen":
                    separator_token = 160000  # or another suitable value that won't conflict with real token IDs
                else:
                    separator_token = 65535
                combined = np.array(prompt + [separator_token] + response, dtype=binary_builder._dtype)
                binary_builder.add_item(torch.from_numpy(combined))

            json_file.write(json.dumps({
                "instruction": MATH_INSTRUCTION,
                "prompt": prompt_str,
                "input": line["question"],
                "output": line["answer"],
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid}/{len(valid_examples)} documents. {inst_num} instances.",
                    f"({lid/elapsed:.2f} docs/s, {mbs:.2f} MB/s).")

        binary_builder.finalize(idx_file)
        json_file.close()
                
        print("\nFinal Statistics:")
        print(f"Total examples processed: {inst_num}")
        print(f"Prompt lengths - Mean: {np.mean(prompt_lens):.1f}, Max: {np.max(prompt_lens)}, Min: {np.min(prompt_lens)}")
        print(f"Response lengths - Mean: {np.mean(response_lens):.1f}, Max: {np.max(response_lens)}, Min: {np.min(response_lens)}")

        # Add after writing the binary file
        # Verify the written data
        data = np.fromfile(bin_file, dtype=np.uint32 if args.model_type=="qwen" else np.uint16)
        print(f"Written data stats:")
        print(f"Max token ID: {data.max()}")
        print(f"Min token ID: {data.min()}")
        print(f"Data shape: {data.shape}")


if __name__ == '__main__':
    main()