#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=100g
#SBATCH -t 0

set -x  # echo commands to stdout
set -e  # exit on error
module load singularity
singularity shell /projects/tir1/singularity/tensorflow-ubuntu-16.04.2-lts-nvidia-375.26.img

module load cuda-8.0 cudnn-8.0-5.1
export CUDA_HOME="/projects/tir1/cuda-8.0.27.1"
PATH=$HOME/bin:$PATH
PATH=$CUDA_HOME/bin:$PATH
export PATH  
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH
export CPATH=/opt/cudnn-8.0/include:$CPATH
export LIBRARY_PATH=/opt/cudnn-8.0/lib64:$LD_LIBRARY_PATH

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
NHID=256
SEED=10
BSZ=16
EMSIZE=200
DROP=0.5
NL=1
LR=0.002

DATA_DIR="../music_data/CMaj_Nottingham"
PRETRAIN_DIR="../music_data/010_011_large/" 
TMP_PREFIX="010_011_nott_large"
EPOCHS=3
VOCAB_PATHS=\[\"$DATA_DIR\",\"$PRETRAIN_DIR\"\]
SAVE="../tmp/mrnn_nott_pretrain.pt"
python train.py --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$PRETRAIN_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --temperature=0.5 --save=$SAVE --baseline --synth_data

EPOCHS=50
TMP_PREFIX="CMaj_Nottingham"
python train.py --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --temperature=0.5 --save=$SAVE --baseline --checkpoint=$PRETRAIN
python train.py --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --use_metaf --train_info_out=$2 --temperature=0.5 --save=$SAVE --mode='generate' --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --checkpoint=$SAVE --outf='mrnn'


