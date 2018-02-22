
VOCABF="../tmp/cmaj_nott_sv"
CORPUSF="../tmp/cmaj_nott_sv_corpus"
# DATA_DIR="../music_data/CMaj_Nottingham/" 
DATA_DIR="../music_data/debug_data_small/" 
CONDITION_NOTES=8
WINDOW=8
C=2
DISTANCE_THRESHOLD=1
TEMPERATURE=1.0
ARCH='crnn'
RNN_TYPE='GRU'
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE".pt"
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"
NHID=4
SEED=1

rm -rf ../tmp/*
python train.py --save=$FILE_NAME --epochs=60 --nhid=$NHID --data=$DATA_DIR --vocabf=$VOCABF --corpusf=$CORPUSF --emsize=10 --batch_size=2 --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --nhid=1024 --seed=$SEED --epochs=10

