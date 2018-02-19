#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=100g
#SBATCH -t 0
#SBATCH -o ../../slurm-out/cond_torch_2_1.txt

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
OUTF="cond"
DATA_DIR="../music_data/CMaj_Nottingham/" 
# DATA_DIR="../music_data/debug_data_small/" 
CONDITION_NOTES=8
C=2
DISTANCE_THRESHOLD=1
# TODO REMEMBER TO CHANGE THE FILE NAME ABOVE AND BELOW.  
FILE_NAME="../tmp/cond_"$C"_"$DISTANCE_THRESHOLD".pt"
TEMPERATURE=1.0
# NO PROGRESS TOKENS IF YOU WANT TO DO CRNN
python train.py --cuda --save=$FILE_NAME --epochs=60 --nhid=1024 --data=$DATA_DIR --vocabf=$VOCABF --corpusf=$CORPUSF --emsize=10 --batch_size=64 --c=$C --distance_threshold=$DISTANCE_THRESHOLD --arch='crnn' --measure_tokens 
# python old_generate.py --cuda --outf=$OUTF --checkpoint=$FILE_NAME --vocabf=$VOCABF --corpusf=$CORPUSF --num_out=10 --condition_piece="../music_data/CMaj_Nottingham/valid/jigs_simple_chords_16.mid" --condition_notes=$CONDITION_NOTES --arch='crnn' --window=$WINDOW --distance_threshold=$DISTANCE_THRESHOLD --temperature=$TEMPERATURE
