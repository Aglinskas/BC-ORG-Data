#!/bin/bash

#SBATCH --job-name=job-NDAR-fMRIprep
#SBATCH --output=./logs/output-NDAR-fMRIprep-%a
#SBATCH --error=./logs/error-NDAR-fMRIprep-%a

#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --mem=32gb
#SBATCH --array=1-50

#rm -rf fmriprep_ABIDEII-SDSU_1.slurm;nano fmriprep_ABIDEII-SDSU_1.slurm
#sbatch fmriprep_ABIDEII-SDSU_1.slurm
#$SLURM_ARRAY_TASK_ID


module load singularity/

data_dir=$HOME'/ds-2075'
output_dir=$HOME'/fprep-ds-2075'

sing_image=$HOME/fmriprep.simg
fs_lic=$HOME'/fs_licence.txt'
sub=sub-`printf %03d $SLURM_ARRAY_TASK_ID`



echo $HOME
echo $data_dir
echo $output_dir
echo $sub


singularity run --bind /data/pijarj/scratch:/scratch --cleanenv $sing_image $data_dir $output_dir participant --participant-label $sub --fs-no-reconall --task-id 'rest' --fs-license-file $fs_lic --ignore slicetiming

###fmriprep ds-2075 fprep-ds-2075 participant --participant-label sub-001 --fs-no-reconall --task-id 'rest' --fs-license-file fs_licence.txt

