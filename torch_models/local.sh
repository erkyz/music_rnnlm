
# DATA_DIR="../music_data/debug_data_small/" 
# DATA_DIR="../music_data/CMaj_Nottingham/" 
TMP_PREFIX="../tmp/cmaj_jigs"
DATA_DIR="../music_data/CMaj_Jigs/" 
CONDITION_NOTES=20
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH='cell'
RNN_TYPE='GRU'
NHID=128
SEED=$1
BSZ=52
EMSIZE=100
DROP=0.2
NL=1
EPOCHS=30
LR=0.001
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE".pt"
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_h"$NHID"_e"$EMSIZE"

# MAKE SURE BOTH ARE most_recent OR NOT
# python train.py --save=$FILE_NAME --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=0 --nlayers=$NL --epochs=$EPOCHS #--most_recent 
# python old_generate.py --arch=$ARCH --outf=$ARCH --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --condition_notes=$CONDITION_NOTES --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C # --most_recent 
python get_hiddens.py --arch=$ARCH --outf=$ARCH --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --condition_notes=$CONDITION_NOTES --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent
