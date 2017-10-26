#!/bin/bash

set -x  # echo commands to stdout
set -e  # exit on error
module load singularity
singularity shell /projects/tir1/singularity/tensorflow-ubuntu-16.04.2-lts-nvidia-375.26.img

module load cuda-8.0 cudnn-8.0-5.1
#module load cudnn-8.0-5.1
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
python generate.py --cuda --outf="base.mid" --checkpoint="../tmp/base.pt"
python train.py --cuda --bptt=400 --save="../tmp/bptt70.pt"
python generate.py --cuda --outf="bptt70.mid" --checkpoint="../tmp/bptt70.pt"
python train.py --cuda --nlayers=4 --save="../tmp/nlayer4.pt"
python generate.py --cuda --outf="nlayer4.mid" --checkpoint="../tmp/nlayer4.pt"
python train.py --cuda --nhid=400 --save="../tmp/nhid400.pt"
python generate.py --cuda --outf="nhid400.mid" --checkpoint="../tmp/nhid400.pt"
python train.py --cuda --emsize=20 --save="../tmp/emsize20.pt"
python generate.py --cuda --outf="emsize20.mid" --checkpoint="../tmp/emsize20.pt"

