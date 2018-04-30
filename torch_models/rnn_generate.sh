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
BSZ=4
EMSIZE=200
DROP=0.5
NL=1
EPOCHS=50

# DATA_DIR="../music_data/ashover_010_011_large/" 
# TMP_PREFIX="010_011_large"
# SAVE='../tmp/010_011_large.pt'

DATA_DIR="../music_data/CMaj_Nottingham/"
TMP_PREFIX="CMaj_Nottingham_segment"
SAVE="../tmp/nott_mrnn_segment.pt"
# SAVE="../tmp/cell_1024_nott.pt"
METAF="segment_meta.p"
OUTF="GRU_1024"

# DATA_DIR="../music_data/ashover/" 
VOCAB_PATHS=\[\"$DATA_DIR\"\]

LR=0.002
# python train.py --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf=$METAF --train_info_out=$2 --temperature=0.5 --save=$SAVE --baseline # --synth_data
python train.py --cuda --nhid=$NHID --vocab_paths=$VOCAB_PATHS --path=$DATA_DIR --tmp_prefix=$TMP_PREFIX --batch_size=$BSZ --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch=$ARCH --rnn_type=$RNN_TYPE --seed=$SEED --lr=$LR --emsize=$EMSIZE --dropout=$DROP --skip_first_n_note_losses=$SKIP --nlayers=$NL --epochs=$EPOCHS --metaf=$METAF --train_info_out=$2 --temperature=0.5 --save=$SAVE --mode='generate' --condition_piece="../music_data/CMaj_Nottingham/train/jigs_simple_chords_90.mid" --checkpoint=$SAVE --outf=$OUTF


