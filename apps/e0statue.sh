#! /bin/bash

#SBATCH --job-name="e0Statue"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arnau.coiduras@city.ac.uk
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --output job%J.output
#SBATCH --error e0statue.err
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --partition=normal


module load cuda/10.0
python3 train_shape.py