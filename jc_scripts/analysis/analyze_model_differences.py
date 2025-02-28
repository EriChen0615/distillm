import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict
import pandas as pd
from tabulate import tabulate

def load_models_and_tokenizers():
    """Load both models and tokenizers."""
    model_7b_name = "Qwen/Qwen2.5-7B-Instruct"
    model_05b_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    print("Loading tokenizers...")
    tokenizer_7b = AutoTokenizer.from_pretrained(model_7b_name)
    tokenizer_05b = AutoTokenizer.from_pretrained(model_05b_name)
    
    print("Loading models...")
    model_7b = AutoModelForCausalLM.from_pretrained(model_7b_name, torch_dtype=torch.float16)
    model_05b = AutoModelForCausalLM.from_pretrained(model_05b_name, torch_dtype=torch.float16)
    
    return model_7b, model_05b, tokenizer_7b, tokenizer_05b

def analyze_vocab_differences(tokenizer_7b, tokenizer_05b):
    """Analyze vocabulary differences between tokenizers."""
    vocab_7b = set(tokenizer_7b.get_vocab().keys())
    vocab_05b = set(tokenizer_05b.get_vocab().keys())
    
    print("\nVocabulary Analysis:")
    print(f"7B vocab size: {len(vocab_7b)}")
    print(f"0.5B vocab size: {len(vocab_05b)}")
    print(f"Tokens in 7B but not in 0.5B: {len(vocab_7b - vocab_05b)}")
    print(f"Tokens in 0.5B but not in 7B: {len(vocab_05b - vocab_7b)}")
    
    # Sample some unique tokens from 7B
    unique_to_7b = list(vocab_7b - vocab_05b)[:10]
    print("\nSample tokens unique to 7B:")
    for token in unique_to_7b:
        print(f"  {token}")

def analyze_model_dimensions(model_7b, model_05b):
    """Analyze model output dimensions."""
    print("\nModel Dimensions:")
    print(f"7B output dimension: {model_7b.config.vocab_size}")
    print(f"0.5B output dimension: {model_05b.config.vocab_size}")

def analyze_token_mapping(tokenizer_7b, tokenizer_05b):
    """Analyze how tokens are mapped between models using example texts."""
    example_texts = [
        "Hello world!",
        "This is a complex example with numbers 123 and symbols @#$",
        "Let's try some technical terms: neural networks, transformers, and deep learning",
        "Here's some math: f(x) = ax² + bx + c",
        "And some code: def hello_world(): print('Hello!')"
    ]
    
    results = []
    for text in example_texts:
        # Tokenize with both tokenizers
        tokens_7b = tokenizer_7b.tokenize(text)
        tokens_05b = tokenizer_05b.tokenize(text)
        
        # Create comparison
        result = {
            'Text': text,
            '7B Tokens': tokens_7b,
            '0.5B Tokens': tokens_05b,
            'Length Diff': len(tokens_7b) - len(tokens_05b)
        }
        results.append(result)
    
    # Print results in a nice table
    print("\nTokenization Comparison:")
    for result in results:
        print("\nInput text:", result['Text'])
        print("7B tokens:", result['7B Tokens'])
        print("0.5B tokens:", result['0.5B Tokens'])
        print(f"Length difference: {result['Length Diff']}")
        print("-" * 80)

def main():
    # Load models and tokenizers
    model_7b, model_05b, tokenizer_7b, tokenizer_05b = load_models_and_tokenizers()
    
    # Run analyses
    analyze_vocab_differences(tokenizer_7b, tokenizer_05b)
    analyze_model_dimensions(model_7b, model_05b)
    analyze_token_mapping(tokenizer_7b, tokenizer_05b)
    
    # Additional analysis: check embedding dimensions
    print("\nModel Architecture Details:")
    print(f"7B hidden size: {model_7b.config.hidden_size}")
    print(f"0.5B hidden size: {model_05b.config.hidden_size}")

if __name__ == "__main__":
    main() 