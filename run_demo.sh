#!/bin/bash


#SBATCH -o logs/slurm_pyscript_%j.job
#SBATCH -e logs/slurm_error_%j.job
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:1
##SBATCH --constraint=a100_80gb
#SBATCH -t 02:00:00
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate patchsae

cd /lustre/groups/labs/marr/qscd01/workspace/furkan.dasdelen/patchsae-dev-main

PYTHONPATH=./ nohup python -u demo.py > logs/demo_wbc.txt
