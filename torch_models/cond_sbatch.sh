#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=100g
#SBATCH -t 0
#SBATCH -o ../../slurm-out/cond_torch_vine.txt

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
