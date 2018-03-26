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


# TMP_PREFIX="cmaj_nott"
# DATA_DIR="../music_data/CMaj_Nottingham/" 
TMP_PREFIX="cmaj_jigs"
DATA_DIR="../music_data/CMaj_Jigs/" 
SKIP=30
CONDITION_NOTES=30
C=1.5
DISTANCE_THRESHOLD=0
TEMPERATURE=1.0
ARCH='xrnn'
RNN_TYPE='GRU'
NHID=64
SEED=$1
BSZ=52
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=200
LR=0.0005
OUTF=$ARCH"_nh"$NHID
VANILLA_FNAME='../tmp/vanilla2.pt'

python train.py --save=$VANILLA_FNAME --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='cell' --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --most_recent --cuda
python train.py --arch='cell' --checkpoint=$VANILLA_FNAME --cuda --tmp_prefix=$TMP_PREFIX --condition_piece="../music_data/CMaj_Jigs/train/jigs_simple_chords_90.mid" --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent --mode='generate' --condition_notes=$CONDITION_NOTES

# python train.py --nhid=$NHID --data=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --most_recent --cuda --vanilla_ckpt=$VANILLA_FNAME
# python old_generate.py --arch=$ARCH --outf=$ARCH"_nh"$NHID --checkpoint=$FILE_NAME --tmp_prefix=$TMP_PREFIX --num_out=5 --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --condition_notes=$CONDITION_NOTES --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent --vanilla_ckpt=$VANILLA_FNAME --cuda

# NOTE THAT THIS IS CELL
# python train.py --arch='cell' --checkpoint=$VANILLA_FNAME --tmp_prefix=$TMP_PREFIX --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE --c=$C --most_recent --cuda --mode='get_hiddens'

