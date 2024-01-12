import os
from os.path import join as opj

resnet = 34
epochs = 30
nclasses = 1
# for i in range(9):
for i in [8]:
    print(f"training split{i}")
    d = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/MRT_organoide/2022_batch_lightfield_microscopy/new_folder_structure/resized/cyst_classification_bin_mri_paper_revision/splits'
    d2 = opj(d, 'imgs', f'split{i}')
    outdir = opj(*[d, 'results', f'split{i}'])
    os.makedirs(outdir, exist_ok=True)
    train_dir = opj(d2, 'training')
    val_dir = opj(d2, 'validation')
    test_dir = opj(d2, 'test')
    cmd = f'CUDA_VISIBLE_DEVICES=0 python resnet_classifier.py {resnet} {nclasses} {epochs} {train_dir} {val_dir} -ts {test_dir} -g 1 -tr -s {outdir} --ce_weights 1.0 1.0 -lr 1e-6 -tb_outdir {outdir}'
    print(cmd)
    os.system(cmd)
    print('#'*50)
