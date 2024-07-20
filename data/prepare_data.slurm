#!/usr/bin/env bash
#SBATCH --job-name preparedata
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=70
#SBATCH --mem=600G
#SBATCH --time=14-00:00:00
#SBATCH --output=../logs/preaparedata_%A_%a.out
#SBATCH --partition=compute
#SBATCH --array=1-1%1


source /home/kc2819/.bashrc
conda deactivate
conda activate chotallm


if [[ ${SLURM_ARRAY_TASK_ID} == 1 ]] ; then
    python openwebtext/prepare_data.py --dataset openwebtext-10k --encoder gpt2  --num_proc 5 --test_size 0.0005 --samples 2 --save_dir openwebtext/splits
fi
