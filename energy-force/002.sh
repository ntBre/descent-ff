#!/bin/bash
#SBATCH -J smee
#SBATCH -p standard
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=72gb
#SBATCH --account dmobley_lab
#SBATCH --export ALL
#SBATCH --mail-user=bwestbr1@uci.edu
#SBATCH --constraint=fastscratch
#SBATCH -o logs/2.out

date
hostname

source ~/.bashrc
mamba activate fb-196-qcnew

python 002-parameterize.py
       
date
