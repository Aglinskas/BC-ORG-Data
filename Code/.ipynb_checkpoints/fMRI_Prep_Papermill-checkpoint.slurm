#!/bin/bash
#SBATCH --job-name=fMRI_Prep_papermill_notebook
#SBATCH --output=fMRI_Prep_Output_papermill_notebook
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=10:00:00
#SBATCH --mail-user=pijarj@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=16gb

#rm -rf slurm_papermill_notebook_GPU_gpu100.slurm; nano slurm_papermill_notebook_GPU_gpu100.slurm
#sbatch slurm_papermill_notebook_GPU_gpu100.slurm

notebook_name='fMRI_Prep_Tutorial.ipynb'

outname='fMRI_Prep_Papermill_2.ipynb'

pwd

papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar





