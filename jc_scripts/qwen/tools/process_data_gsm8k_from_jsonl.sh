BASE_PATH=${1:-"."}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_gsm8k_from_jsonl.py \
    --data-dir ${BASE_PATH}/data/gsm8k/train \
    --processed-data-dir ${BASE_PATH}/processed_data/gsm8k/qwen2.5/prompt \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 256 \
    --only-prompt \
    --model-type qwen

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_gsm8k_from_jsonl.py \
    --data-dir ${BASE_PATH}/data/gsm8k/train \
    --processed-data-dir ${BASE_PATH}/processed_data/gsm8k/qwen2.5/full \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num 256 \
    --model-type qwen
