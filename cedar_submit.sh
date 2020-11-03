#!/bin/bash
#SBATCH --nodes=1 
#SBATCH --gres=gpu:p100l:4   
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24    # There are 24 CPU cores on P100 Cedar GPU nodes
#SBATCH --mem=0               # Request the full memory of the node
#SBATCH --time=3:00
#SBATCH --account=def-someuser
hostname
nvidia-smi