#!/usr/bin/env bash
#SBATCH --job-name fullblown_transformer
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time=14-00:00:00
#SBATCH --output=./logs/fullblown_transformer_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=1-1%1


source /home/kc2819/.bashrc
conda deactivate
module load cudatoolkit/10.2 cudnn/cuda-10.2/7.6.5
conda activate chotallm

nvidia-smi


if [[ ${SLURM_ARRAY_TASK_ID} == 1 ]] ; then
    python src/fullblown_bigram/bigram_MHA.py
fi
