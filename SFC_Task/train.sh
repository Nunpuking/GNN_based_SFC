RUNNING_MODE='train'
MODEL_NAME='GG_RNN'
LEARNING_FASHION=$2
PREDICT_MODE=$3
ENVIRONMENT='Simulation'

TOPOLOGY_CHANGE_MODE=$7
DEPLOYMENT_CHANGE_MODE=$8

SAVE_DIR='./results/'
LOAD_DIR='./results/' #'./backup/'

DATA_SPLIT=0
LOAD_CHECKPOINT=0

# !!! you should be careful to 'load_pt_model' variable
LOAD_PT_MODEL='' #'./results/SL_SFC_VNFLevel_RMSprop_0.0001LR_1Batch/model0.pth.best.pth'

N_RUNNING=1
EPOCHS=10000
PRINT_ITER=20
VALID_ITER=1000
BATCH_SIZE=$4

PATIENCE=10
LR_DECAY=3
LR_DECAY_RATIO=0.1
LR=$6 #SL:1e-4, RL:1e-5
OPT=$5 # RMSprop

# Only for RL
RL_EPSILON=0.1

if [ $PREDICT_MODE == 'NodeLevel' ]
then
    DELAY_COEFF=10
elif [ $PREDICT_MODE == 'VNFLevel' ]
then
    DELAY_COEFF=10
fi

DISCOUNT_FACTOR=0.999

GRU_STEPS=5
NODE_STATE_DIM=128
POSENC_NODE_DIM=4
MAX_N_NODES=50
RECURRENT_DELAY=0.1
ADJ_TEMPERATURE=2

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZTc1ODBhNDktMWE1My00NGI1LTk4YjQtMzVhMTIzYjI2MjRiIn0="

CUDA_VISIBLE_DEVICES=$1 python3 run.py\
    --running_mode=$RUNNING_MODE --model_name=$MODEL_NAME --learning_fashion=$LEARNING_FASHION\
    --predict_mode=$PREDICT_MODE --environment=$ENVIRONMENT\
    --topology_change_mode=$TOPOLOGY_CHANGE_MODE --deployment_change_mode=$DEPLOYMENT_CHANGE_MODE\
    --save_dir=$SAVE_DIR --load_dir=$LOAD_DIR --data_split=$DATA_SPLIT --n_running=$N_RUNNING\
    --epochs=$EPOCHS --print_iter=$PRINT_ITER --valid_iter=$VALID_ITER --patience=$PATIENCE\
    --lr_decay=$LR_DECAY --lr_decay_ratio=$LR_DECAY_RATIO --lr=$LR --opt=$OPT\
    --rl_epsilon=$RL_EPSILON --delay_coeff=$DELAY_COEFF --discount_factor=$DISCOUNT_FACTOR\
    --GRU_steps=$GRU_STEPS --node_state_dim=$NODE_STATE_DIM --posenc_node_dim=$POSENC_NODE_DIM\
    --max_n_nodes=$MAX_N_NODES --recurrent_delay=$RECURRENT_DELAY --adj_temperature=$ADJ_TEMPERATURE\
    --batch_size=$BATCH_SIZE --load_checkpoint=$LOAD_CHECKPOINT --load_pt_model=$LOAD_PT_MODEL
