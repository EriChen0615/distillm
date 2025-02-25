#! /bin/bash
#SBATCH -A BYRNE-SL2-GPU
#SBATCH -J generate_data_seqkd
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
MASTER_PORT=${2-2113}
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
CKPT_NAME="Qwen/Qwen2.5-3B-Instruct-LoRASFT"
# CKPT="${BASE_PATH}/results/gpt2/train/sft/gpt2-xlarge/"
# CKPT="${BASE_PATH}/results/gpt2-xlarge/train/sft/e10-bs2-lr5e-05-G1-N1-NN1/70050/"
CKPT="../../hf_models/Qwen/Qwen2.5-3B-Instruct"
# CKPT="${BASE_PATH}/results/gpt2-xlarge/train/sft/e10-bs2-lr5e-05-G1-N1-NN1/70050/"
# data
DATA_DIR="${BASE_PATH}/processed_data/gsm8k/qwen2.5/full/qwen/"
# hp
EVAL_BATCH_SIZE=16
# runtime
SAVE_PATH="${BASE_PATH}/results/qwen2.5-3B-Instruct-LoRASFT/gen/"


OPTS=""
# model
OPTS+=" --base-path ${BASE_PATH}"
OPTS+=" --model-path ${CKPT}"
OPTS+=" --ckpt-name ${CKPT_NAME}"
OPTS+=" --n-gpu ${GPUS_PER_NODE}"
OPTS+=" --peft lora"
OPTS+=" --peft-path ${BASE_PATH}/results/qwen2.5-3B-Instruct/gsm8k/train/sft/e3-bs4-lr5e-06-G16-N1-NN1-lora-8-32-0.1/336"
# data
OPTS+=" --data-dir ${DATA_DIR}"
OPTS+=" --data-names dolly"
OPTS+=" --num-workers 0"
OPTS+=" --gen-num -1"
OPTS+=" --data-process-workers -1"
OPTS+=" --json-data"
# hp
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --max-length 512"
OPTS+=" --max-prompt-length 256"
# runtime
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --seed-ppo 42"
OPTS+=" --seed 10"
# deepspeed
# OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${BASE_PATH}/configs/deepspeed/ds_config.json"
OPTS+=" --type gen"
# gen
OPTS+=" --do-sample"
OPTS+=" --top-k 0"
OPTS+=" --top-p 1.0"
OPTS+=" --temperature 1.0"


export TOKENIZERS_PARALLELISM=false
export PYTHONIOENCODING=utf-8
export PYTHONPATH=${BASE_PATH}
CMD="torchrun ${DISTRIBUTED_ARGS} ${BASE_PATH}/generate.py ${OPTS} $@"


echo ${CMD}
echo "PYTHONPATH=${PYTHONPATH}"
mkdir -p ${SAVE_PATH}
${CMD}
