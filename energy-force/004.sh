#!/bin/bash
#SBATCH -J smee-004
#SBATCH -p standard
#SBATCH -t 288:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120gb
#SBATCH --account dmobley_lab
#SBATCH --export ALL
#SBATCH --mail-user=bwestbr1@uci.edu
#SBATCH --constraint=fastscratch

date
hostname

source ~/.bashrc
mamba activate fb-196-qcnew

python 004-train.py $SLURM_CPUS_PER_TASK

date
