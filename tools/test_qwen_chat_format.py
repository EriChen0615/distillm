from transformers import AutoTokenizer
import json

MATH_INSTRUCTION = """You are a helpful math assistant. Your task is to solve math word problems step by step.
Show your work and explain your reasoning clearly.
Make sure your final answer is correct and clearly stated."""

def main():
    # Load Qwen tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Example question from GSM8K
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest to her neighbors for $2 per egg. How much money does she make per day?"
    
    # Our current manual format
    manual_format = (
        "<|im_start|>Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Question:\n{input}\n\n### Response:\n<|im_end|><|im_start|>Assistant:"
    ).format(instruction=MATH_INSTRUCTION, input=question)
    
    # Using Qwen's chat template
    messages = [
        {"role": "system", "content": MATH_INSTRUCTION},
        {"role": "user", "content": question}
    ]
    chat_format = tokenizer.apply_chat_template(messages, tokenize=False)
    
    print("\n=== Manual Format ===")
    print(manual_format)
    print("\n=== Chat Template Format ===")
    print(chat_format)
    
    # Compare tokenization
    manual_tokens = tokenizer.encode(manual_format)
    chat_tokens = tokenizer.encode(chat_format)
    
    print("\n=== Token Comparison ===")
    print(f"Manual format tokens: {len(manual_tokens)}")
    print(f"Chat template tokens: {len(chat_tokens)}")
    print("\nTokens match:", manual_tokens == chat_tokens)
    
    if manual_tokens != chat_tokens:
        print("\nDecoded Manual Format:")
        print(tokenizer.decode(manual_tokens))
        print("\nDecoded Chat Format:")
        print(tokenizer.decode(chat_tokens))
        
        # Save both formats to file for easier comparison
        with open("format_comparison.json", "w") as f:
            json.dump({
                "manual_format": manual_format,
                "chat_format": chat_format,
                "manual_tokens": manual_tokens,
                "chat_tokens": chat_tokens,
                "manual_decoded": tokenizer.decode(manual_tokens),
                "chat_decoded": tokenizer.decode(chat_tokens)
            }, f, indent=2)

if __name__ == "__main__":
    main() 