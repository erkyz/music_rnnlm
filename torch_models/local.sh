
# DATA_DIR="../music_data/CMaj_Nottingham/" 
# TMP_PREFIX="../tmp/cmaj_jigs"
# DATA_DIR="../music_data/CMaj_Jigs/" 
TMP_PREFIX="test"
DATA_DIR="../music_data/ashover[[0,1,0],[0,0,1]]/" 
CONDITION_NOTES=6
SKIP=0
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=128
SEED=10
BSZ=4
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=75
LR=0.0005
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"
VANILLA_FILE_NAME="vanilla"

# MAKE SURE BOTH ARE most_recent OR NOT
# python train.py --save=$VANILLA_FILE_NAME --nhid=$NHID --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='cell' --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS 

# rm -rf ../tmp/test_sv.p
# rm -rf ../tmp/test_corpus.p
python train.py --nhid=$NHID --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --baseline
# python train.py --nhid=$NHID --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --mode='generate' --condition_piece="../music_data/ashover[[0,1,0],[0,0,1]]/train/[0, 0, 1]3.mid" --outf="TEST"
