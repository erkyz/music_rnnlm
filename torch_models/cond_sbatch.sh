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

# DATA_DIR="../music_data/debug_data_small/" 
# DATA_DIR="../music_data/CMaj_Nottingham/" 
TMP_PREFIX="../tmp/cmaj_jigs"
DATA_DIR="../music_data/CMaj_Jigs/" 
CONDITION_NOTES=20
C=2
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH=$1
RNN_TYPE='GRU'
NHID=128
SEED=10
BSZ=52
EMSIZE=100
DROP=0.5
NL=1
EPOCHS=100
LR=0.001
# FILE_NAME="../tmp/vine_87.pt"
FILE_NAME="../tmp/"$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_drop"$DROP"_nh"$NHID"_em"$EMSIZE".pt"
OUTF="$ARCH"_"$C"_"$DISTANCE_THRESHOLD"_"$RNN_TYPE"_h"$NHID"_e"$EMSIZE"

# MAKE SURE BOTH ARE most_recent OR NOT
# python train.py --save=$FILE_NAME --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=0 --nlayers=$NL --epochs=$EPOCHS --cuda # --most_recent 
# python old_generate.py --arch=$ARCH --outf=$ARCH"nh"$NHID --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --condition_notes=$CONDITION_NOTES --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --cuda # --most_recent 
python get_hiddens.py --arch=$ARCH --outf=$ARCH"nh"$NHID --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --condition_notes=$CONDITION_NOTES --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --cuda # --most_recent 






