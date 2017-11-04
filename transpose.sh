#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH -t 0
#SBATCH -o ../slurm-out/transpose_valid.txt

python transpose.py --dir=valid
