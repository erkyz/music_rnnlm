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
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=$4 # TODO
SEED=10
BSZ=4 # TODO pay attention to this!
EMSIZE=200
DROP=$5
NL=$6

EPOCHS=30
#DATA="../music_data/CMaj_Nottingham/"
#TMP_PREFIX="CMaj_Nottingham_correct"
DATA="../music_data/guitar_melodies_1500_final/"
TMP_PREFIX="guitar_1500"

#DATA="../music_data/010_011_large/" 
#TMP_PREFIX="010_011_large"
INFO_OUT=$2
SAVE=$3
METAF="meta.p"
LR=0.002
VOCAB_PATHS=\[\"$DATA\"\]

python train.py --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf=$METAF --train_info_out=$INFO_OUT --temperature=0.5 --save=$SAVE --baseline --input_feed #--checkpoint=$3 # --cnn_encoder  # --synth_data


