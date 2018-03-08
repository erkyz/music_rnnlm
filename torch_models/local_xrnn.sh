
# DATA_DIR="../music_data/CMaj_Nottingham/" 
# TMP_PREFIX="../tmp/cmaj_jigs"
# DATA_DIR="../music_data/CMaj_Jigs/" 
TMP_PREFIX="../tmp/cmaj_small"
DATA_DIR="../music_data/debug_data_small/" 
CONDITION_NOTES=20
SKIP=0
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=128
SEED=1000
BSZ=1 # TODO
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=300
LR=0.0005
# FILE_NAME="../tmp/vine_87.pt"
VANILLA_FILE_NAME="../tmp/vanilla_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_h"$NHID"_e"$EMSIZE"

# MAKE SURE BOTH ARE most_recent OR NOT
# python train.py --save=$VANILLA_FILE_NAME --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='cell' --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --most_recent 
python train.py --save='../test.pt' --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='cell' --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --most_recent 
# python train.py --save=$FILE_NAME --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --most_recent --vanilla_ckpt=$VANILLA_FILE_NAME
# python old_generate.py --arch=$ARCH --outf=$ARCH"_nh"$NHID --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --condition_notes=$CONDITION_NOTES --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent 
# python get_hiddens.py --arch=$ARCH --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent

