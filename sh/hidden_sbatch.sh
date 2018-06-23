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

DATA_DIR="../music_data/CMaj_Jigs/" 
TMP_PREFIX="../tmp/cmaj_jigs"
CONDITION_NOTES=8
WINDOW=8
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=128
SEED=$2
BSZ=52
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=200
SKIP=0
LR=0.001
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"
VANILLA_FILE_NAME="../tmp/vanilla_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE"_skip$SKIP.pt"

python get_hiddens.py --cuda --checkpoint=$FILE_NAME --vanilla_ckpt=$VANILLA_FILE_NAME --tmp_prefix=$TMP_PREFIX --num_out=10 --condition_piece="../music_data/CMaj_Nottingham/valid/jigs_simple_chords_256.mid" --window=$WINDOW --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --cuda --arch=$ARCH

