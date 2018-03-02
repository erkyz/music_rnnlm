
VOCABF="../tmp/cmaj_nott_sv"
CORPUSF="../tmp/cmaj_nott_sv_corpus"
# DATA_DIR="../music_data/debug_data_small/" 
DATA_DIR="../music_data/CMaj_Nottingham/" 
CONDITION_NOTES=2
WINDOW=8
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH='xrnn'
RNN_TYPE='GRU'
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE".pt"
# FILE_NAME="../tmp/vine_87.pt"
NHID=128
SEED=$1
BSZ=1
EMSIZE=100
LR=20
DROP=0
NL=1
EPOCHS=400
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_h"$NHID"_e"$EMSIZE"

# MAKE SURE BOTH ARE most_recent OR NOT
rm -rf ../tmp/*
python train.py --save=$FILE_NAME --nhid=$NHID --data=$DATA_DIR --vocabf=$VOCABF --corpusf=$CORPUSF --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --most_recent --skip_first_n_note_losses=0 --nlayers=$NL --epochs=$EPOCHS
python old_generate.py --arch=$ARCH --outf=$ARCH --checkpoint=$FILE_NAME --vocabf=$VOCABF --corpusf=$CORPUSF --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/xmas_simple_chords_1.mid" --condition_notes=$CONDITION_NOTES --window=$WINDOW --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent 
# python get_hiddens.py --arch=$ARCH --outf=$ARCH --checkpoint=$FILE_NAME --vocabf=$VOCABF --corpusf=$CORPUSF --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/xmas_simple_chords_1.mid" --condition_notes=$CONDITION_NOTES --window=$WINDOW --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE 
