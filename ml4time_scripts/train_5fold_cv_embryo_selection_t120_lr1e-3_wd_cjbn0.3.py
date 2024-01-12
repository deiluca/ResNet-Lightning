import os
from os.path import join as opj

epochs = 200
nclasses = 3
resd = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/images_t120/results_lr0.001_wd1e-4_cj-bn0.3'
d = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/images_t120'
cj_bn = 0.3
wd = 1e-4
for resnet in [34]:
    for lr in [1e-3]:
        for i in range(5):
            print(f"training split{i}")
            outdir = opj(*[resd, f'results_epochs_resnet{resnet}_lr{lr}_wd{wd}_ch-bn{cj_bn}', f'split{i}'])
            os.makedirs(outdir, exist_ok=True)
            train_dir = opj(d, f'split{i}_training')
            val_dir = opj(d, f'split{i}_validation')
            test_dir = opj(d, f'split{i}_test')
            cmd = f'CUDA_VISIBLE_DEVICES=0 python resnet_classifier.py {resnet} {nclasses} {epochs} {train_dir} {val_dir} -ts {test_dir} -g 1 -tr -s {outdir} --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805 -lr {lr} -tb_outdir {outdir} --cj_bn {cj_bn} --weight_decay {wd}'
            print(cmd)
            os.system(cmd)
            print('#'*50)
        