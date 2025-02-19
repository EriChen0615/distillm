BASE_PATH=${1:-"."}

export TF_CPP_MIN_LOG_LEVEL=3

# only prompt for MiniLLM train
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_gsm8k.py \
    --data-dir "openai/gsm8k" \
    --processed-data-dir ${BASE_PATH}/processed_data/qwen2.5/gsm8k/prompt \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --only-prompt \
    --model-type qwen \
    --dev-num 256 \

# prompt and response for baselines
PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_gsm8k.py \
    --data-dir "openai/gsm8k" \
    --processed-data-dir ${BASE_PATH}/processed_data/qwen2.5/gsm8k/full \
    --model-path Qwen/Qwen2.5-1.5B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --model-type qwen \
    --dev-num 256 \
