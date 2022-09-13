{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "405e308f-bd88-4d9a-a9da-52afcaf08a67",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mon Sep  5 13:57:42 EDT 2022\n"
     ]
    }
   ],
   "source": [
    "!date"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "81b8d8b0-e400-4313-866f-bc4456cc5901",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'/mmfs1/data/pijarj/BC-ORG-Data/Code'"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pwd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "d015a7b7-3a67-45a8-923b-b82322b37522",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/mmfs1/data/pijarj/BC-ORG-Data/Data\n"
     ]
    }
   ],
   "source": [
    "cd ../Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "d5a45cf0-6591-4ac8-ad7a-10ccc30e4b2b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "CPU times: user 1.81 s, sys: 1.1 s, total: 2.9 s\n",
      "Wall time: 35.8 s\n"
     ]
    }
   ],
   "source": [
    "%%time\n",
    "import pandas as pd\n",
    "import os\n",
    "import shutil\n",
    "import numpy as np\n",
    "from tqdm import tqdm\n",
    "import ants\n",
    "import sys\n",
    "import nibabel as nib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "426f3691-550a-4fef-8c3b-b3e3ec5d7188",
   "metadata": {},
   "outputs": [],
   "source": [
    "#make txt file into csv\n",
    "read_file = pd.read_csv ('/mmfs1/data/pijarj/NDAR_BoldAnat10/image03.txt',delimiter='\\t',low_memory=False)\n",
    "read_file.to_csv ('/mmfs1/data/pijarj/BC-ORG-Data/Data/image03.csv', index=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "fc43eaa4-2e70-4c3a-8fea-ad6af88da935",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drops non nii or nii.gz types\n",
    "df = pd.read_csv('image03.csv',low_memory=False)\n",
    "df = df.iloc[1::] # First row is just another index, let's drop it\n",
    "df['image_file'] = df['image_file'].astype(str) # convert entries to strings (so nan become 'nan')\n",
    "\n",
    "df['is_nii_gz'] = [str(file).endswith('.nii.gz') for file in df['image_file'].values]\n",
    "df['is_nii'] = [str(file).endswith('.nii') for file in df['image_file'].values]\n",
    "\n",
    "# only keep the niftis \n",
    "df = df.iloc[df['is_nii_gz'].values + df['is_nii'].values]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bdd28e6a-3ef5-4f1e-816d-986a534a9015",
   "metadata": {},
   "outputs": [],
   "source": [
    "args = sys.argv\n",
    "script_name = sys.argv[0]\n",
    "study_id = sys.argv[1]\n",
    "subject_id = sys.argv[2]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68671351-c4d3-494d-b387-fff083e646c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f'name of this script is: {script_name}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d207a01-4fcb-4d07-869d-0b2b61d5ee87",
   "metadata": {},
   "outputs": [],
   "source": [
    "#making new folder and getting study id\n",
    "print(f'study name is: {study_id}')\n",
    "#os.mkdir('ds-{subject_id}')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0d4f60bc-2c90-41f9-94d9-819106c05fb7",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "def safe_mkdir(path):\n",
    "    if not os.path.exists(path):\n",
    "        os.mkdir('ds-{subject_id}')\n",
    "    else:\n",
    "        pass"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "4f2b8f7b-104d-438b-baf7-5285a5bcd905",
   "metadata": {
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "name of this script is: /data/pijarj/anaconda3/lib/python3.8/site-packages/ipykernel_launcher.py\n",
      "study name is: -f\n",
      "subject number is: /mmfs1/data/pijarj/.local/share/jupyter/runtime/kernel-822f65e5-eb39-47b9-9d60-52790d2bfa7b.json\n"
     ]
    }
   ],
   "source": [
    "#make directory\n",
    "ndar_root = '/mmfs1/data/pijarj/NDAR_BoldAnat10/' \n",
    "bids_root = '/mmfs1/data/pijarj/'\n",
    "safe_mkdir(os.path.join(bids_root,f'DS{col_id}.csv'))\n",
    "for s in tqdm(range(1,nsubjects+1)):\n",
    "    safe_mkdir(os.path.join(bids_root,f'ds-{col_id}',f'sub-{s:03d}'))\n",
    "    safe_mkdir(os.path.join(bids_root,f'ds-{col_id}',f'sub-{s:03d}','func'))\n",
    "    safe_mkdir(os.path.join(bids_root,f'ds-{col_id}',f'sub-{s:03d}','anat'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "302de894-5238-4f49-88b8-51c7f4a92bbd",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "100%|██████████| 192/192 [00:00<00:00, 961.85it/s]\n"
     ]
    }
   ],
   "source": [
    "#making local paths column\n",
    "#df = pd.read_csv(f'DS{col_id}.csv')\n",
    "n = len(study_df)\n",
    "#root = './image03/'\n",
    "root = os.path.expanduser('~/NDAR_BoldAnat10/image03/')\n",
    "local_paths = list()\n",
    "for i in tqdm(range(n)):\n",
    "    s3_path = study_df['image_file'].values[i]\n",
    "    splits = s3_path.split('/')[4::]\n",
    "    relative = '/'.join(splits)\n",
    "    local_path = os.path.join(root,relative)\n",
    "    assert os.path.exists(os.path.join(root,relative))\n",
    "    local_paths.append(local_path)\n",
    "study_df['local_paths'] = local_paths"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea78f667-7407-4d81-91a8-11e6034c9a14",
   "metadata": {},
   "outputs": [],
   "source": [
    "#copying over subject\n",
    "print(f'subject number is: {subject_id}')\n",
    "sub = use_subjects[subject_id]\n",
    "sub_df = study_df.iloc[study_df['subjectkey'].values==sub]\n",
    "fmri_idx = sub_df['scan_type'].values=='fMRI'\n",
    "anat_idx = sub_df['scan_type'].isin(['MR structural (T1)','MR structural (MPRAGE)']).values\n",
    "epi_fn = sub_df.iloc[fmri_idx]['local_paths'].values[0]\n",
    "anat_fn = sub_df.iloc[anat_idx]['local_paths'].values[0]\n",
    "epi_path = os.path.join(ndar_root,epi_fn)\n",
    "anat_path = os.path.join(ndar_root,anat_fn)\n",
    "epi_dest = os.path.join(bids_root,f'ds-{col_id}',f'sub-{s+1:03d}','func',f'sub-{s+1:03d}_task-rest_bold.nii.gz')\n",
    "anat_dest = os.path.join(bids_root,f'ds-{col_id}',f'sub-{s+1:03d}','anat',f'sub-{s+1:03d}_T1w.nii.gz')\n",
    "t1 = ants.image_read(anat_path) \n",
    "bold = ants.image_read(epi_path)\n",
    "t1.to_filename(anat_dest)\n",
    "bold.to_filename(epi_dest)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "eea4b3e3-2269-4f81-bb82-dde328f30cbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "def write_json(data,filepath):\n",
    "    import json\n",
    "    with open(filepath, 'w') as outfile:\n",
    "        json.dump(data, outfile)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0469c9e0-ddf5-4a1d-b3e2-4cfa9c439717",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "bold_json = {\"RepetitionTime\" : bold.spacing[-1],\n",
    "         \"TaskName\" : 'rest'}\n",
    "write_json(bold_json,epi_dest.replace('.nii.gz','.json')) \n",
    "im = nib.load(epi_dest)\n",
    "header = im.header.copy()\n",
    "header.set_xyzt_units(xyz='mm', t='sec')\n",
    "nib.nifti1.Nifti1Image(im.get_fdata(), None, header=header).to_filename(epi_dest)\n",
    "assert nib.load(epi_dest).header.get_xyzt_units()==('mm', 'sec'),'timing missing from header'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5ba92b96-8e5d-4759-9513-081ba1d509c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "data = {\n",
    "    \"Name\" : study_df[\"collection_title\"].values[0] ,\n",
    "    \"RepetitionTime\": 2.0,\n",
    "    \"SliceTiming\" : 2.0 ,\n",
    "    \"TaskName\" : \"taskrest\" ,\n",
    "    \"BIDSVersion\" : \"20.2.0\"}\n",
    "json_string = json.dumps(data)\n",
    "print(json_string)\n",
    "with open(os.path.join(bids_root,f'ds-{col_id}','dataset_description.json'), 'w') as outfile:\n",
    "    json.dump(json_string, outfile)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
