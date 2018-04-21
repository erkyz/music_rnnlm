
# DATA_DIR="../music_data/CMaj_Nottingham/" 
# TMP_PREFIX="../tmp/cmaj_jigs"
# DATA_DIR="../music_data/CMaj_Jigs/" 
# DATA_DIR="../music_data/ashover[[0,0]]/" 
CONDITION_NOTES=0
SKIP=0
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=256
SEED=10
BSZ=4 # TODO
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=75
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"

TMP_PREFIX="ashover"
DATA_DIR="../music_data/ashover[[0,1,0],[0,0,1]]/"
ASHOVER="../music_data/ashover/" 
VOCAB_PATHS=\[\"$DATA_DIR\",\"$ASHOVER\"\]
VANILLA_FILE_NAME="vanilla"
LR=0.005
# python train.py --save=$VANILLA_FILE_NAME --nhid=$NHID --path=$ASHOVER --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='cell' --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --vocab_paths=$VOCAB_PATHS


# rm -rf ../tmp/test_one_sv.p
# rm -rf ../tmp/test_one_corpus.p
LR=0.001
TMP_PREFIX="test"
SAVE="../tmp/pretrain.pt"
# python train.py --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --temperature=0.5 --baseline --save=$SAVE
TMP_PREFIX="ashover"
# python train.py --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$ASHOVER --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --vanilla_ckpt=$VANILLA_FILE_NAME --train_info_out=$2 --temperature=0.5 --baseline --checkpoint=$SAVE --save=$SAVE

# --condition_piece="../music_data/ashover[[0,1,0],[0,0,1]]/train/[0, 0, 1]3.mid"
# python train.py --nhid=$NHID --path=$ASHOVER --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ  --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --mode='generate' --condition_piece="../music_data/ashover/train/ashover_simple_chords_3.mid" --outf="TEST" --temperature=0.5 --vanilla_ckpt=$VANILLA_FILE_NAME --checkpoint=$SAVE
# python train.py --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$ASHOVER --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --vanilla_ckpt=$VANILLA_FILE_NAME --train_info_out=$2 --temperature=0.5 --baseline --checkpoint=$SAVE --save=$SAVE --mode='generate' --condition_piece="../music_data/ashover/train/ashover_simple_chords_3.mid" --outf="TEST" 
