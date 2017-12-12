#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem=15g
#SBATCH -t 0
#SBATCH -o ../../slurm-out/hrnn_torch.txt
#TODO

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

FILE_NAME="../tmp/hrnn.pt"
ARCH="hrnn"
VOCABF="../tmp/cmaj_nott_sv"
CORPUSF="../tmp/cmaj_nott_sv_corpus"
CONDITION_MEASURES=3
# python train.py --cuda --save=$FILE_NAME --epochs=60 --nhid=256 --data="../music_data/CMaj_Nottingham/" --vocabf=$VOCABF --corpusf=$CORPUSF --arch=$ARCH  --emsize=10 --batch_size=64 --factorize --progress_tokens
python generate.py --cuda --outf="condhrnn" --checkpoint=$FILE_NAME --vocabf=$VOCABF --corpusf=$CORPUSF --num_out=10 --condition_piece="../music_data/CMaj_Nottingham/valid/reels_simple_chords_322.mid" --condition_measures=$CONDITION_MEASURES --arch=$ARCH --factorize --progress_tokens


