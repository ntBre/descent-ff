#!/bin/bash

config=${1?no config file supplied}

sbatch <<INP
#!/bin/bash
#SBATCH -J smee
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
#SBATCH -o logs/`date +%Y-%m-%d_%H-%M-%S`.out

date
hostname

source ~/.bashrc
mamba activate fb-196-qcnew

python main.py $config

date
INP
