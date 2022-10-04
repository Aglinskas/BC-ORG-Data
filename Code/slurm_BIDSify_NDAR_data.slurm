#!/bin/bash

#SBATCH --job-name=job_bidsify_NDAR
#SBATCH --output=./slurm_files/output_bidsify_NDAR_%a.txt
#SBATCH --error=./slurm_files/error_bidsify_NDAR_%a.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --mem=16gb
#SBATCH --array=0-52


study_id=2338
sub=$SLURM_ARRAY_TASK_ID

python BIDSify_NDAR_data.py $study_id $sub



