#! /bin/bash

MASTER_ADDR=localhost
MASTER_PORT=${2-2012}
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${3-1}

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# model
BASE_PATH=${1-"."}
CKPT_NAME="Qwen/Qwen2.5-0.5B-Instruct"
# CKPT="${CKPT_NAME}/"
CKPT="../../hf_models/${CKPT_NAME}"
#CKPT="${BASE_PATH}/checkpoints/${CKPT_NAME}/"
# CKPT="gpt2" # download automatically
# data
DATA_DIR="${BASE_PATH}/processed_data/gsm8k/qwen2.5/full/qwen/"
# hp
BATCH_SIZE=8
LR=0.000005
GRAD_ACC=8
EVAL_BATCH_SIZE=16
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --num-workers 0"
# OPTS+=" --dev-num 256"
OPTS+=" --dev-num 256"
# hp
OPTS+=" --lr ${LR}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --warmup-iters 0"
OPTS+=" --lr-decay-style cosine"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --epochs 20"
# length
OPTS+=" --max-length ${MAX_LENGTH}"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --eval-gen"
OPTS+=" --save-interval -1"
OPTS+=" --eval-interval -1"
OPTS+=" --log-interval 4"
OPTS+=" --mid-log-num -1"
OPTS+=" --save ${SAVE_PATH}"
# seed
OPTS+=" --seed ${SEED}"
# deepspeed
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type lm"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
export WANDB_DISABLED=False
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
export WANDB_PROJECT="gsm8k-expo"
export WANDB_NAME="gsm8k_qwen2.5-0.5B/sft-lr-${LR}-bs-${BATCH_SIZE}-grad-${GRAD_ACC}-eval-bs-${EVAL_BATCH_SIZE}-max-length-${MAX_LENGTH}-max-prompt-length-${MAX_PROMPT_LENGTH}-seed-${SEED}"
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
