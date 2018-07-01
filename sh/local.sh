# HYPERPARAMS
SEED=10
CONDITION_NOTES=0
SKIP=0
DISTANCE_THRESHOLD=0
RNN_TYPE='GRU'
NHID=128
BSZ=4 
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=20
LR=0.001
INPUT_FEED=0

# COMMAND LINE INPUTS
ARCH=$1
TRAIN_INFO_OUT=$2   # where to save stdout from training
GEN_OUTF=$3         # file prefix for generated songs. Note that "generated/" dir must exist.
DATA_DIR=$4         # example: DATA_DIR="music_data/CMaj_Nottingham" WITHOUT the final slash 
VOCAB_PATHS=\[\"$DATA_DIR\"\]

# TRAIN
python main.py --mode="train" --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf='meta.p' --train_info_out=$TRAIN_INFO_OUT  --input_feed_num=$INPUT_FEED 

# GENERATE
TEMPERATURE=1.0
python main.py --mode="generate" --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf='meta.p' --input_feed_num=$INPUT_FEED --temperature=$TEMPERATURE --outf=$GEN_OUTF

