RUNNING_MODE='test'
MODEL_NAME='GG_RNN'
LEARNING_FASHION=$2
PREDICT_MODE=$3
ENVIRONMENT='Simulation'

TOPOLOGY_CHANGE_MODE_TEST=0
DEPLOYMENT_CHANGE_MODE_TEST=0

SAVE_DIR='./results/'
LOAD_DIR='./results/' #'./backup/'

N_RUNNING=1
BATCH_SIZE=$4

LR=$6
OPT=$5

# Only for RL
RL_EPSILON=0
DELAY_COEFF=0
DISCOUNT_FACTOR=0.999

GRU_STEPS=5
NODE_STATE_DIM=128
POSENC_NODE_DIM=4
MAX_N_NODES=50
RECURRENT_DELAY=0.1
ADJ_TEMPERATURE=2

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiZTc1ODBhNDktMWE1My00NGI1LTk4YjQtMzVhMTIzYjI2MjRiIn0="

CUDA_VISIBLE_DEVICES=$1 python3.8 run.py\
    --running_mode=$RUNNING_MODE --model_name=$MODEL_NAME --learning_fashion=$LEARNING_FASHION\
    --predict_mode=$PREDICT_MODE --environment=$ENVIRONMENT\
    --topology_change_mode_test=$TOPOLOGY_CHANGE_MODE_TEST\
    --deployment_change_mode_test=$DEPLOYMENT_CHANGE_MODE_TEST\
    --save_dir=$SAVE_DIR --load_dir=$LOAD_DIR --n_running=$N_RUNNING\
    --rl_epsilon=$RL_EPSILON --delay_coeff=$DELAY_COEFF --discount_factor=$DISCOUNT_FACTOR\
    --GRU_steps=$GRU_STEPS --node_state_dim=$NODE_STATE_DIM --posenc_node_dim=$POSENC_NODE_DIM\
    --max_n_nodes=$MAX_N_NODES --recurrent_delay=$RECURRENT_DELAY --adj_temperature=$ADJ_TEMPERATURE\
    --batch_size=$BATCH_SIZE --lr=$LR --opt=$OPT
