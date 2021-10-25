#!/bin/bash

#SBATCH --job-name=massbias                            # Job name
#SBATCH --ntasks=40                                   # Number of CPUs
#SBATCH --time=1000:00:00                              # Time limit hrs:min:sec
#SBATCH --output=results_analysis-ens.log            # Standard output and error log
#SBATCH --exclude=gpu01                               # Don't hog the GPU node


# memory
ulimit -s unlimited

# location
cd /home/jmframe/mclstm_2021_mass_balance/results
pwd

# load conda environment
source /home/jmframe/.bashrc
conda activate mc-paper

# run code
python -u results_analysis-ens.py
