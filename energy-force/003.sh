#!/bin/bash
#SBATCH -J smee-003
#SBATCH -p standard
#SBATCH -t 72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=72gb
#SBATCH --account dmobley_lab
#SBATCH --export ALL
#SBATCH --mail-user=bwestbr1@uci.edu
#SBATCH --constraint=fastscratch

date
hostname

echo requested $SLURM_CPUS_PER_TASK CPUs

source ~/.bashrc
mamba activate descent-ff

python 003-cluster-and-filter.py -c -n $SLURM_CPUS_PER_TASK
       
date
