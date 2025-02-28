#! /bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH -J distillm-base
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
CKPT_NAME="Qwen/Qwen2.5-0.5B-Instruct-SFT-3ep"
# CKPT="../../hf_models/${CKPT_NAME}"
CKPT="${BASE_PATH}/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e3-bs8-lr1e-05-G8-N1-NN1/336"
# PEFT_PATH="${BASE_PATH}/results/qwen2.5-0.5B-Instruct/gsm8k/train/sft/e3-bs8-lr1e-05-G8-N1-NN1/336"
# PEFT_NAME="Qwen2.5-0.5B-Instruct-LoRASFT-3ep"
# CKPT="${BASE_PATH}/results/gpt2/train/init/${CKPT_NAME}"
TEACHER_CKPT="${BASE_PATH}/../../hf_models/Qwen/Qwen2.5-7B-Instruct"
TEACHER_CKPT_NAME="Qwen2.5-7B-Instruct-LoRASFT"
TEACHER_PEFT_NAME="Qwen2.5-7B-Instruct-LoRASFT"
# TEACHER_PEFT_PATH="third_party/distillm/results/qwen2.5-3B-Instruct/gsm8k/train/sft/e3-bs4-lr5e-06-G16-N1-NN1-lora-8-32-0.1/336"
TEACHER_PEFT_PATH="${BASE_PATH}/results/qwen2.5-7B-Instruct/gsm8k/train/sft/e10-bs1-lr5e-06-G64-N1-NN1-lora-8-32-0.1/1120"
# data
DATA_DIR="${BASE_PATH}/processed_data/gsm8k/qwen2.5/pseudo/qwen/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/qwen2.5/512/1M/"
# hp
BATCH_SIZE=1
LR=0.00001
GRAD_ACC=64
EVAL_BATCH_SIZE=16
EPOCHS=10
# length
MAX_LENGTH=512
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2.5-0.5B-Instruct/gsm8k/train/distillm/7B_teacher_10ep-no_ptloss"
# seed
SEED=10


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --teacher-peft-name ${TEACHER_PEFT_NAME}"
OPTS+=" --teacher-peft-path ${TEACHER_PEFT_PATH}"
OPTS+=" --teacher-model-fp16"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --model-type qwen"
# OPTS+=" --peft-path ${PEFT_PATH}"
# OPTS+=" --peft-name ${PEFT_NAME}"
# data
OPTS+=" --data-dir ${DATA_DIR}"
# OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --num-workers 4"
OPTS+=" --dev-num 256"
# OPTS+=" --dev-num 16"
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
OPTS+=" --type adaptive-sfkl"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# distillm
OPTS+=" --student-gen"
OPTS+=" --gen-num-beams 1"
OPTS+=" --gen-top-p 1.0"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"
OPTS+=" --capacity 1000"


export NCCL_DEBUG=""
# export WANDB_DISABLED=False
export WANDB_NAME="gsm8k_qwen2.5-0.5B/distillm_from_init_teacher=7B-LoRASFT-10ep-no_ptloss-lr-${LR}-bs-${BATCH_SIZE}-grad-${GRAD_ACC}-max-length-${MAX_LENGTH}-max-prompt-length-${MAX_PROMPT_LENGTH}-seed-${SEED}-ep-${EPOCHS}"
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/finetune.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
CODE_BASE=HF ${CMD}
