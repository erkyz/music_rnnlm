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

# COMMAND LINE INPUTS
ARCH=$1
TRAIN_INFO_OUT=$2   # where to save stdout from training
DATA_DIR=$3         # example: DATA_DIR="music_data/CMaj_Nottingham" WITHOUT the final slash 
VOCAB_PATHS=\[\"$DATA_DIR\"\]

# TRAIN
python main.py --mode="train" --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf='meta.p' --train_info_out=$TRAIN_INFO_OUT  --input_feed_num=$INPUT_FEED 

# GENERATE
TEMPERATURE=1.0
python main.py --mode="generate" --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --batch_size=$BSZ --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf='meta.p' --input_feed_num=$INPUT_FEED --temperature=$TEMPERATURE

