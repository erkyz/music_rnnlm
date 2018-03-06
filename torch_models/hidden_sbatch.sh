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

DATA_DIR="../music_data/CMaj_Nottingham/" 
TMP_PREFIX="../tmp/cmaj_nott"
# DATA_DIR="../music_data/debug_data_small/" 
CONDITION_NOTES=8
WINDOW=8
C=2
DISTANCE_THRESHOLD=1
TEMPERATURE=1.0
ARCH='cell'
RNN_TYPE='GRU'
FILE_NAME="../tmp/$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE.pt"
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"
# python train.py --cuda --save=$FILE_NAME --epochs=60 --nhid=1024 --data=$DATA_DIR --vocabf=$VOCABF --corpusf=$CORPUSF --emsize=10 --batch_size=64 --c=$C --distance_threshold=$DISTANCE_THRESHOLD --measure_tokens --arch=$ARCH --rnn_type=$RNN_TYPE
python get_hiddens.py --cuda --outf=$OUTF --checkpoint=$FILE_NAME --vocabf=$VOCABF --corpusf=$CORPUSF --num_out=10 --condition_piece="../music_data/CMaj_Nottingham/train/xmas_simple_chords_1.mid" --condition_notes=$CONDITION_NOTES --window=$WINDOW --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE
