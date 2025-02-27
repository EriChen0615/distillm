BASE_PATH=${1-"."}

MAX_LENGTH=512

PYTHONPATH=${BASE_PATH} python3 ${BASE_PATH}/tools/process_data_pretrain.py \
    --data-dir ${BASE_PATH}/data/openwebtext \
    --processed-data-dir ${BASE_PATH}/processed_data/openwebtext/qwen2.5/${MAX_LENGTH}/ \
    --model-path ${BASE_PATH}/../../hf_models/Qwen/Qwen2.5-0.5B-Instruct \
    --max-length ${MAX_LENGTH} \
    --train-num 1000000 \
    --data-process-workers 32 \
    --dev-num 1000 \
    --model-type qwen
    # --train-num 1000000 \