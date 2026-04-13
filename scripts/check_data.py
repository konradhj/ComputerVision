"""Quick script to check volume dimensions across institutions."""
import nibabel as nib
import os

data_root = '/cluster/projects/vc/courses/TDT17/mic/ODELIA2025/data/'

for inst in ['CAM', 'MHA', 'RSH', 'RUMC', 'UKA']:
    inst_dir = os.path.join(data_root, inst, 'data_unilateral')
    if not os.path.exists(inst_dir):
        print(f'{inst}: directory not found')
        continue
    sample = sorted(os.listdir(inst_dir))[0]
    path = os.path.join(inst_dir, sample, 'Pre.nii.gz')
    if not os.path.exists(path):
        print(f'{inst}: Pre.nii.gz not found')
        continue
    img = nib.load(path)
    print(f'{inst} ({sample}): shape={img.shape}, voxel_size={img.header.get_zooms()}')
