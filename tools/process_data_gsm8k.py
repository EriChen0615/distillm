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
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

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
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Verify token IDs are valid
        if max(prompt_tokens) >= len(Encoder.tokenizer):
            print(f"Max token ID in prompt: {max(prompt_tokens)}")
            print(f"Vocab size: {len(Encoder.tokenizer)}")
            raise ValueError(f"Token ID {max(prompt_tokens)} exceeds vocab size {len(Encoder.tokenizer)}")
        
        full_tokens = Encoder.tokenizer.encode(prompt + response, add_special_tokens=False) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens):]
        
        if len(prompt_tokens) > self.args.max_prompt_length:
            prompt_tokens = prompt_tokens[:self.args.max_prompt_length]
            # return None, None, None, None, len(line)
        
        if self.args.model_type == "qwen":
            separator_token = 4294967295  # or another suitable value that won't conflict with real token IDs
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
        
        inst_num = 0
        print("#"*10, split, "#"*10)
        
        prompt_lens = []
        response_lens = []
        
        json_file = open(os.path.join(args.processed_data_dir, f"{split}.jsonl"), "w")
        
        for lid, example in enumerate(all_data[split]):
            if DEBUG_MODE:
                print(f"\nProcessing example {lid}:")
                print("Raw example:", example)
            
            line, prompt_str, prompt, response, bytes_processed = encoder.encode(example)
            
            if DEBUG_MODE:
                print("\nProcessed data:")
                print("Formatted prompt:", prompt_str)
                print("Prompt tokens:", prompt[:50], "..." if len(prompt) > 50 else "")
                print("Response tokens:", response[:50], "..." if len(response) > 50 else "")
            
            total_bytes_processed += bytes_processed
            if prompt is None:
                if DEBUG_MODE:
                    print("Skipping example - prompt is None")
                continue
            
            if args.only_prompt:
                if len(prompt) < args.max_length:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    if DEBUG_MODE:
                        print("Skipping example - prompt too long")
                    continue
            else:
                if args.model_type == "qwen":
                    separator_token = 160000  # or another suitable value that won't conflict with real token IDs
                else:
                    separator_token = 65535
                binary_builder.add_item(torch.IntTensor(prompt + [separator_token] + response))

            json_file.write(json.dumps({
                "instruction": MATH_INSTRUCTION,
                "prompt": prompt_str,
                "input": line["question"],
                "output": line["answer"],
            }) + "\n")

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:  # Changed back to 1000 for production
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed:.2f} docs/s, {mbs:.2f} MB/s).")
            
            # Debug pause only in debug mode
            if DEBUG_MODE and lid == 5:
                print("\nFirst 5 examples processed. Check the output above.")
                user_input = input("Press Enter to continue, or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break

        binary_builder.finalize(idx_file)
        json_file.close()
                
        print("\nFinal Statistics:")
        print("Total examples processed:", len(prompt_lens))
        print("Prompt lengths - Mean:", np.mean(prompt_lens), "Max:", np.max(prompt_lens), "Min:", np.min(prompt_lens))
        print("Response lengths - Mean:", np.mean(response_lens), "Max:", np.max(response_lens), "Min:", np.min(response_lens))

        # Add after writing the binary file
        # Verify the written data
        data = np.fromfile(bin_file, dtype=np.uint32 if args.model_type=="qwen" else np.uint16)
        print(f"Written data stats:")
        print(f"Max token ID: {data.max()}")
        print(f"Min token ID: {data.min()}")
        print(f"Data shape: {data.shape}")


if __name__ == '__main__':
    main()