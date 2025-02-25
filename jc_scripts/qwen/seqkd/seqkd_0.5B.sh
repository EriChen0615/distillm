#! /bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH -J seqkd_base
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue
#SBATCH -p ampere

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
CKPT="../../hf_models/${CKPT_NAME}"

TEACHER_CKPT="${BASE_PATH}/../../hf_models/Qwen/Qwen2.5-3B-Instruct"
TEACHER_CKPT_NAME="Qwen2.5-3B-Instruct-LoRASFT"
TEACHER_PEFT_NAME="Qwen2.5-3B-Instruct-LoRASFT"
TEACHER_PEFT_PATH="third_party/distillm/results/qwen2.5-3B-Instruct/gsm8k/train/sft/e3-bs4-lr5e-06-G16-N1-NN1-lora-8-32-0.1/336"
# data
DATA_DIR="${BASE_PATH}/processed_data/gsm8k/qwen2.5/pseudo/qwen/"
# LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/10M/"
# hp
BATCH_SIZE=4
LR=0.00001
GRAD_ACC=16
EVAL_BATCH_SIZE=16
EPOCHS=10
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2.5-0.5B-Instruct/gsm8k/train/seqkd"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# data
OPTS+=" --data-dir ${DATA_DIR}"
# OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
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
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --kd-ratio 1.0"
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
# OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
# type
OPTS+=" --type kd"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export NCCL_DEBUG=""
# export WANDB_DISABLED=True
export WANDB_NAME="gsm8k_qwen2.5-0.5B/seqkd-lr-${LR}-bs-${BATCH_SIZE}-grad-${GRAD_ACC}-eval-bs-${EVAL_BATCH_SIZE}-max-length-${MAX_LENGTH}-max-prompt-length-${MAX_PROMPT_LENGTH}-seed-${SEED}-ep-${EPOCHS}"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
