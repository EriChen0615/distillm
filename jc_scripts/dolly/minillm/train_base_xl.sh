#! /bin/bash
#SBATCH -J Dolly-GPT2-base-MiniLLM
#SBATCH -A BYRNE-SL2-GPU
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
CKPT_NAME="base-init"
# CKPT="${BASE_PATH}/results/gpt2-base/train/init/gpt2-base"
CKPT="${BASE_PATH}/results/gpt2-base/train/sft-init/e3-bs8-lr0.0005-G1-N1-NN1/5253"
TEACHER_CKPT_NAME="xlarge-sft"
TEACHER_CKPT="${BASE_PATH}/results/gpt2-xlarge/train/sft/e10-bs2-lr5e-05-G1-N1-NN1/70050"

# data
PROMPT_DATA_DIR="${BASE_PATH}/processed_data/dolly/prompt/gpt2/"
LM_DATA_DIR="${BASE_PATH}/processed_data/openwebtext/gpt2/512/10M/"
# runtime
SAVE_PATH="${BASE_PATH}/results/gpt2-base/train/minillm_241218/"
# hp
GRAD_ACC=1
BATCH_SIZE=8
CHUNK_SIZE=16


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --teacher-model-path ${TEACHER_CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --teacher-ckpt-name ${TEACHER_CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --n-nodes ${NNODES}"
OPTS+=" --teacher-model-fp16"
# OPTS+=" --gradient-checkpointing"
# data
OPTS+=" --prompt-data-dir ${PROMPT_DATA_DIR}"
OPTS+=" --lm-data-dir ${LM_DATA_DIR}"
OPTS+=" --dev-num 1000"
OPTS+=" --num-workers 0"
# hp
OPTS+=" --epochs 10"
OPTS+=" --total-iters 5000"
OPTS+=" --kd-ratio 0.5"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --lr 5e-6"
OPTS+=" --lr-min 5e-6"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
OPTS+=" --warmup-iters 100"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed 10"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed-lm 7"
OPTS+=" --save-interval 500"
OPTS+=" --eval-interval 100"
OPTS+=" --log-interval 16"
OPTS+=" --mid-log-num 1"
# ppo
OPTS+=" --type minillm"
OPTS+=" --ppo-epochs 4"
OPTS+=" --num-rollouts 256"
OPTS+=" --chunk-size ${CHUNK_SIZE}"
# minillm
OPTS+=" --length-norm"
OPTS+=" --single-step-reg"
OPTS+=" --teacher-mixed-alpha 0.2"
# reward
OPTS+=" --reward-scaling 0.5"
OPTS+=" --cliprange-reward 100"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"
# deepspeed
# OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"

export NCCL_DEBUG=""
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_NAME="dolly/gpt2/minillm-train_0.1B_1.5B"
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/train_minillm.py ${OPTS} $@"

echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
