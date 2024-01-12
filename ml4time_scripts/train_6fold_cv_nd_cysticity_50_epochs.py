import os
from os.path import join as opj

resnet = 34
epochs = 50
nclasses = 1
for i in range(6):
    print(f"training split{i}")
    d = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/MRT_organoide/datensatz_neuer_batch_13_04_23/mri_new_batch_microscopy_brightfield_images/NeuerBatch/all_images_resnetlight_cysticity_classification/run_22_06_23_nd_cystic_classification'
    d2 = opj(d, f'split{i}')
    outdir = opj(*[d, 'results_50epochs', f'split{i}'])
    os.makedirs(outdir, exist_ok=True)
    train_dir = opj(d2, 'train')
    val_dir = opj(d2, 'val')
    test_dir = opj(d2, 'test')
    cmd = f'CUDA_VISIBLE_DEVICES=0 python resnet_classifier.py {resnet} {nclasses} {epochs} {train_dir} {val_dir} -ts {test_dir} -g 1 -tr -s {outdir} --ce_weights 1.0 7.05555555556 -lr 1e-6 -tb_outdir {outdir}'
    print(cmd)
    os.system(cmd)
    print('#'*50)
    