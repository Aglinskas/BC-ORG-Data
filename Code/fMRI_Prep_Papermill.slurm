#!/bin/bash
#SBATCH --job-name=fMRI_Prep_papermill_notebook
#SBATCH --output=fMRI_Prep_Output_papermill_notebook
#SBATCH --error=fMRI_Prep_Error_papermill_notebook
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mail-user=pijarj@bc.edu
#SBATCH --mail-type=ALL
#SBATCH --mem=32gb

#rm -rf slurm_papermill_notebook_GPU_gpu100.slurm; nano slurm_papermill_notebook_GPU_gpu100.slurm
#sbatch slurm_papermill_notebook_GPU_gpu100.slurm

notebook_name='fMRI_Prep_Tutorial.ipynb'

outname='fMRI_Prep_Papermill_6.ipynb'

pwd

papermill $notebook_name $outname --autosave-cell-every 5 --progress-bar





