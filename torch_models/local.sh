
# DATA_DIR="../music_data/CMaj_Nottingham/" 
# TMP_PREFIX="../tmp/cmaj_jigs"
# DATA_DIR="../music_data/CMaj_Jigs/" 
TMP_PREFIX="test"
DATA_DIR="../music_data/ashover[[0,1,0],[0,0,1]]/" 
CONDITION_NOTES=0
SKIP=0
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=128
SEED=10
BSZ=1 # TODO
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=200
LR=0.0005
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"
VANILLA_FILE_NAME="vanilla"

# MAKE SURE BOTH ARE most_recent OR NOT
# python train.py --save=$VANILLA_FILE_NAME --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='cell' --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS 

# rm -rf ../tmp/test_corpus.p
python train.py --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --vanilla=$VANILLA_FILE_NAME --use_metaf
# OUTF="test1"
# python train.py --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --vanilla=$VANILLA_FILE_NAME --mode='generate' --condition_piece="../music_data/debug_data_small/train/jigs_simple_chords_90.mid" 

