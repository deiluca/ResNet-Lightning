import os
from os.path import join as opj
import numpy as np

epochs = 200
nclasses = 3
resd = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/images_t120/results_lr_tuning_resnet34'
d = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/images_t120'

# [0.0001,
#  0.0002,
#  0.0003,
#  0.0004,
#  0.0005,
#  0.0006,
#  0.0007,
#  0.0008,
#  0.0009,
#  0.001,
#  0.002,
#  0.003,
#  0.004,
#  0.005,
#  0.006,
#  0.007,
#  0.008,
#  0.009,
#  0.01]
lrs = [round(x, 4) for x in np.linspace(0.0001, 0.001, 10).tolist()] + [round(x, 3) for x in np.linspace(0.001, 0.01, 10).tolist()[1:]]

for resnet in [34]:
    for lr in lrs:
        for i in range(5):
            print(f"training split{i}")
            outdir = opj(*[resd, f'results_epochs_resnet{resnet}_lr{lr}', f'split{i}'])
            os.makedirs(outdir, exist_ok=True)
            train_dir = opj(d, f'split{i}_training')
            val_dir = opj(d, f'split{i}_validation')
            test_dir = opj(d, f'split{i}_test')
            cmd = f'CUDA_VISIBLE_DEVICES=0 python resnet_classifier.py {resnet} {nclasses} {epochs} {train_dir} {val_dir} -ts {test_dir} -g 1 -tr -s {outdir} --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805 -lr {lr} -tb_outdir {outdir}'
            print(cmd)
            os.system(cmd)
            print('#'*50)
        