BASE_PATH=${1-"."}

export TF_CPP_MIN_LOG_LEVEL=3

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_gsm8k_from_jsonl.py \
    --data-dir ${BASE_PATH}/results/qwen2.5-3B-Instruct-LoRASFT/gen/Qwen/Qwen2.5-3B-Instruct-LoRASFT/t1.0-l512 \
    --processed-data-dir ${BASE_PATH}/processed_data/gsm8k/qwen2.5/pseudo \
    --model-path ${BASE_PATH}/../../hf_models/Qwen/Qwen2.5-3B-Instruct \
    --data-process-workers 32 \
    --max-prompt-length 256 \
    --dev-num -1 \
    --model-type qwen

cp ${BASE_PATH}/processed_data/gsm8k/qwen2.5/full/qwen/valid_0.bin ${BASE_PATH}/processed_data/gsm8k/qwen2.5/pseudo/qwen/
cp ${BASE_PATH}/processed_data/gsm8k/qwen2.5/full/qwen/valid_0.idx ${BASE_PATH}/processed_data/gsm8k/qwen2.5/pseudo/qwen/
cp ${BASE_PATH}/processed_data/gsm8k/qwen2.5/full/qwen/valid.jsonl ${BASE_PATH}/processed_data/gsm8k/qwen2.5/pseudo/qwen/
