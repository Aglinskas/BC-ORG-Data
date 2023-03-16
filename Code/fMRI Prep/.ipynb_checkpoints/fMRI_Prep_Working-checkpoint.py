import pandas as pd
import os
import shutil
import numpy as np
from tqdm import tqdm
import ants
import sys
import nibabel as nib

read_file = pd.read_csv ('/mmfs1/data/pijarj/NDAR_BoldAnat10/image03.txt',delimiter='\t',low_memory=False)
read_file.to_csv ('/mmfs1/data/pijarj/BC-ORG-Data/Data/image03.csv', index=0)

df = pd.read_csv('image03.csv',low_memory=False)
df = df.iloc[1::] 
df['image_file'] = df['image_file'].astype(str)

df['is_nii_gz'] = [str(file).endswith('.nii.gz') for file in df['image_file'].values]
df['is_nii'] = [str(file).endswith('.nii') for file in df['image_file'].values]

general_df = df.iloc[df['is_nii_gz'].values + df['is_nii'].values]

args = sys.argv
script_name = sys.argv[0]
study_id = sys.argv[1]
subject_id = sys.argv[2] 

print(f'name of this script is: {script_name}')
print(f'study name is: {study_id}')

# def safe_mkdir(path):
#     if not os.path.exists(path):
#         os.mkdir('ds-{subject_id}')
#     else:
#         pass

ndar_root = '/mmfs1/data/pijarj/NDAR_BoldAnat10/'
bids_root = '/mmfs1/data/pijarj/'
#safe_mkdir(os.path.join(bids_root,f'DS{subject_id}.csv'))
# safe_mkdir(os.path.join(bids_root,f'ds-{study_id}',f'sub-{subject_id}'))
# safe_mkdir(os.path.join(bids_root,f'ds-{study_id}',f'sub-{subject_id}','func'))
# safe_mkdir(os.path.join(bids_root,f'ds-{study_id}',f'sub-{subject_id}','anat'))
(os.path.join(bids_root,f'DS{subject_id}.csv'))
(os.path.join(bids_root,f'ds-{study_id}',f'sub-{subject_id}'))
(os.path.join(bids_root,f'ds-{study_id}',f'sub-{subject_id}','func'))
(os.path.join(bids_root,f'ds-{study_id}',f'sub-{subject_id}','anat'))

study_df = pd.read_csv(f'DS{study_id}.csv')
n = len(study_df)
root = os.path.expanduser('~/NDAR_BoldAnat10/image03/')
local_paths = list()
for i in tqdm(range(n)):
    s3_path = study_df['image_file'].values[i]
    splits = s3_path.split('/')[4::]
    relative = '/'.join(splits)
    local_path = os.path.join(root,relative)
    assert os.path.exists(os.path.join(root,relative))
    local_paths.append(local_path)
study_df['local_paths'] = local_paths
study_df.to_csv(f'/mmfs1/data/pijarj/BC-ORG-Data/Data/DS{study_id}.csv', index=0) 

print(f'subject number is: {subject_id}')
study_subjects = np.unique(study_df['subjectkey'].values)
nsubjects = len(study_subjects)
print(f'{nsubjects} unique subjects')

def check_has_anat_and_epi(sub):
    sub_df = study_df.iloc[study_df['subjectkey'].values==sub]
    fmri_idx = sub_df['scan_type'].values=='fMRI'
    anat_idx = sub_df['scan_type'].isin(['MR structural (T1)','MR structural (MPRAGE)']).values
    return fmri_idx.sum()>0 and anat_idx.sum()>0

has_anat_and_epi = np.array([check_has_anat_and_epi(s) for s in study_subjects])

use_subjects = study_subjects[has_anat_and_epi]
nsubjects = len(use_subjects)
print(f'{nsubjects} subjects with anat + fmri')

s = int(subject_id)
sub = use_subjects[s]
sub_df = study_df.iloc[study_df['subjectkey'].values==sub]
fmri_idx = sub_df['scan_type'].values=='fMRI'
anat_idx = sub_df['scan_type'].isin(['MR structural (T1)','MR structural (MPRAGE)']).values
epi_fn = sub_df.iloc[fmri_idx]['local_paths'].values[0]
anat_fn = sub_df.iloc[anat_idx]['local_paths'].values[0]
epi_path = os.path.join(ndar_root,epi_fn)
anat_path = os.path.join(ndar_root,anat_fn)
epi_dest = os.path.join(bids_root,f'ds-{study_id}',f'sub-{s+1:03d}','func',f'sub-{s+1:03d}_task-rest_bold.nii.gz')
anat_dest = os.path.join(bids_root,f'ds-{study_id}',f'sub-{s+1:03d}','anat',f'sub-{s+1:03d}_T1w.nii.gz')
t1 = ants.image_read(anat_path)
bold = ants.image_read(epi_path)
t1.to_filename(anat_dest)
bold.to_filename(epi_dest)

def write_json(data,filepath):
    import json
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile)

bold_json = {"RepetitionTime" : bold.spacing[-1],
         "TaskName" : 'rest'}
write_json(bold_json,epi_dest.replace('.nii.gz','.json')) 
im = nib.load(epi_dest)
header = im.header.copy()
header.set_xyzt_units(xyz='mm', t='sec')
nib.nifti1.Nifti1Image(im.get_fdata(), None, header=header).to_filename(epi_dest)
assert nib.load(epi_dest).header.get_xyzt_units()==('mm', 'sec'),'timing missing from header'

import json
data = {
    "Name" : study_df["collection_title"].values[0] ,
    "RepetitionTime": 2.0,
    "SliceTiming" : 2.0 ,
    "TaskName" : "taskrest" ,
    "BIDSVersion" : "20.2.0"}
json_string = json.dumps(data)
print(json_string)
with open(os.path.join(bids_root,f'ds-{study_id}','dataset_description.json'), 'w') as outfile:
    json.dump(json_string, outfile)
