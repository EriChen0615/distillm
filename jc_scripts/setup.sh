#!/bin/bash
# python tools/get_openwebtext.py

bash scripts/gpt2/tools/process_data_dolly.sh . # Process Dolly Train / Validation Data
# bash scripts/gpt2/tools/process_data_pretrain.sh . # Process OpenWebText Train / Validation Data
# huggingface-cli download openai-community/gpt2-xl --local-dir checkpoints/gpt2-xl
# huggingface-cli download openai-community/gpt2-large --local-dir checkpoints/gpt2-large
# huggingface-cli download openai-community/gpt2-medium --local-dir checkpoints/gpt2-medium
# huggingface-cli download openai-community/gpt2 --local-dir checkpoints/gpt2

