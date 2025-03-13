#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=leonard.waldmann@tum.de
#SBATCH --output=/home/hk-project-pai00028/tum_mhj8661/code/slurm-%j-%x.out

#SBATCH --job-name=normalization
#SBATCH --partition=large
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32       # default: 38
#SBATCH --time=4:00:00

# fastdevrun='--fastdevrun'
# eval="eval.only_eval=True eval.return_all_results=True eval.remove_ckpts=False"

# ---------- HOREKA ------------
export $(cat /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/.env)
# -----------------------------

/home/hk-project-pai00028/tum_mhj8661/miniforge3/envs/eval/bin/python \
    /home/hk-project-pai00028/tum_mhj8661/code/fm-playground/scripts/check_data_normalization.py