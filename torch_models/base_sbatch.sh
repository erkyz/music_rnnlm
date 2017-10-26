#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -t 0
#SBATCH -o ../../slurm-out/base_torch.txt

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

 
python train.py --cuda --save="../tmp/base.pt"
python generate.py --cuda --outf="base" --checkpoint="../tmp/base.pt" 


