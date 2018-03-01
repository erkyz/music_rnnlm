
VOCABF="../tmp/cmaj_nott_sv"
CORPUSF="../tmp/cmaj_nott_sv_corpus"
# DATA_DIR="../music_data/CMaj_Nottingham/" 
DATA_DIR="../music_data/debug_data_small/" 
CONDITION_NOTES=16
WINDOW=8
C=2
DISTANCE_THRESHOLD=1
TEMPERATURE=0.75
ARCH='vine'
RNN_TYPE='GRU'
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE".pt"
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"
NHID=1024
SEED=$1
BSZ=1
EMSIZE=3000
LR=20
DROP=0

# rm -rf ../tmp/*
# python train.py --save=$FILE_NAME --epochs=60 --nhid=$NHID --data=$DATA_DIR --vocabf=$VOCABF --corpusf=$CORPUSF --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --epochs=12 --lr=$LR --emsize=$EMSIZE --dropout=$DROP --most_recent --skip_first_n_note_losses=0
python old_generate.py --arch=$ARCH --outf=$ARCH --checkpoint=$FILE_NAME --vocabf=$VOCABF --corpusf=$CORPUSF --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/xmas_simple_chords_1.mid" --condition_notes=$CONDITION_NOTES --window=$WINDOW --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE
