
# DATA_DIR="../music_data/CMaj_Nottingham/" 
# TMP_PREFIX="../tmp/cmaj_jigs"
# DATA_DIR="../music_data/CMaj_Jigs/" 
CONDITION_NOTES=0
SKIP=0
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=128
SEED=10
BSZ=4 # TODO
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=20
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"

# DATA_DIR="../music_data/ashover[[0,1,0],[0,1,1]]/" 
# TMP_PREFIX="010_011"
# DATA_DIR="../music_data/ashover[[0,1,0],[0,0,1]]/" 
# TMP_PREFIX="010_001"
DATA_DIR="../music_data/CMaj_Nottingham/" 
TMP_PREFIX="CMaj_Nottingham"
VOCAB_PATHS=\[\"$DATA_DIR\"\]

LR=0.001
python train.py --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --temperature=0.5 --save=$SAVE --baseline # --synth_data
# python train.py --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --temperature=0.7 --save=$SAVE --mode='generate' --condition_piece="../music_data/ashover[[0,1,0],[0,0,1]]/train/[0, 0, 1]3.mid" --checkpoint=$SAVE --outf='mrnn'


