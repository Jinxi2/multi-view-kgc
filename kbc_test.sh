#! /bin/bash

#==========
set -e
set -x
set -u
#set -o pipefail
#==========

#==========configs
conf_fp=$1
CUDA=$2
#source $conf_fp

TASK=wn18rr #fb15k237
NUM_VOCAB=41068 #NUM_VOCAB and NUM_RELATIONS must be consistent with vocab.txt file 
NUM_RELATIONS=11

# training hyper-paramters
BATCH_SIZE=1024
LEARNING_RATE=3e-4
EPOCH=400
SOFT_LABEL=0.15
SKIP_STEPS=1000
MAX_SEQ_LEN=20
HIDDEN_DROPOUT_PROB=0.1 #0.1
ATTENTION_PROBS_DROPOUT_PROB=0.1 #0.1

# file paths for training and evaluation 
DATA="./data"
OUTPUT="./2020_0203_output_${TASK}_${EPOCH}_${BATCH_SIZE}_${LEARNING_RATE}_${SOFT_LABEL}_${HIDDEN_DROPOUT_PROB}_${ATTENTION_PROBS_DROPOUT_PROB}"
#OUTPUT="./2020_0203_output_${TASK}_soft0"
#TRAIN_FILE="$DATA/${TASK}/train.coke.txt"
#VALID_FILE="$DATA/${TASK}/valid.coke.txt"
#TEST_FILE="$DATA/${TASK}/test.coke.txt"
TRAIN_FILE="$DATA/${TASK}/tree.train.txt"
VALID_FILE="$DATA/${TASK}/tree.valid.txt"
TEST_FILE="$DATA/${TASK}/tree.valid.txt"
VOCAB_PATH="$DATA/${TASK}/vocab.txt"
TRUE_TRIPLE_PATH="${DATA}/${TASK}/all.txt"
CHECKPOINTS="$OUTPUT/models"
INIT_CHECKPOINTS=$CHECKPOINTS
LOG_FILE="$OUTPUT/train.log"
LOG_EVAL_FILE="$OUTPUT/test.log"

# transformer net config, the follwoing are default configs for all tasks
HIDDEN_SIZE=256
NUM_HIDDEN_LAYERS=6
NUM_ATTENTION_HEADS=4
MAX_POSITION_EMBEDDINS=20
#=========init env
export CUDA_VISIBLE_DEVICES=$CUDA
export FLAGS_sync_nccl_allreduce=1

#modify to your own path
export LD_LIBRARY_PATH=$(pwd)/env/lib/nccl2.3.7_cuda9.0/lib:/home/work/cudnn/cudnn_v7/cuda/lib64:/home/work/cuda-9.0/extras/CUPTI/lib64/:/home/work/cuda-9.0/lib64/:$LD_LIBRARY_PATH


max_step_id=`ls $INIT_CHECKPOINTS | grep "step" | awk -F"_" '{print $NF}' | grep -v "Found"  |sort -n |tail -1`
max_step_id=51000
INIT_CHECKPOINT_STEP=${INIT_CHECKPOINTS}/step_${max_step_id}
echo "init_checkpoints_steps: $max_step_id"


#--init_checkpoint ${INIT_CHECKPOINT}
echo ">> Begin kbc test now, log file: $LOG_EVAL_FILE"
python3 -u ./bin/run.py \
 --dataset $TASK \
 --vocab_size $NUM_VOCAB \
 --num_relations $NUM_RELATIONS \
 --use_cuda true \
 --do_train false \
 --train_file $TRAIN_FILE \
 --checkpoints $CHECKPOINTS \
 --init_checkpoint ${INIT_CHECKPOINT_STEP} \
 --true_triple_path $TRUE_TRIPLE_PATH \
 --max_seq_len $MAX_SEQ_LEN \
 --soft_label $SOFT_LABEL \
 --batch_size $BATCH_SIZE \
 --epoch $EPOCH \
 --learning_rate $LEARNING_RATE \
 --hidden_dropout_prob $HIDDEN_DROPOUT_PROB \
 --attention_probs_dropout_prob $ATTENTION_PROBS_DROPOUT_PROB \
 --skip_steps $SKIP_STEPS \
 --do_predict true \
 --predict_file $TEST_FILE \
 --vocab_path $VOCAB_PATH \
 --hidden_size $HIDDEN_SIZE \
 --num_hidden_layers $NUM_HIDDEN_LAYERS \
 --num_attention_heads $NUM_ATTENTION_HEADS \
 --max_position_embeddings $MAX_POSITION_EMBEDDINS \
 --use_ema false > $LOG_EVAL_FILE 2>&1

echo ">> Finish kbc test, log file: $LOG_EVAL_FILE"
