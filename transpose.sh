#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:0
#SBATCH --mem=10g
#SBATCH -t 0
#SBATCH -o ../slurm-out/transpose.txt

python3 transpose.py --dir=valid --corpus='guitar'
python3 transpose.py --dir=train --corpus='guitar'


