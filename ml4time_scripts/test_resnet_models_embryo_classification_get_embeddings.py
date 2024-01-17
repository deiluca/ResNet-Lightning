import os
from os.path import join as opj

epochs = 200
nclasses = 3
d = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/Paolo_selected_t_396_corr'
resd = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/Paolo_selected_t_396_corr/results/resnet18_2d_epochs400_lr0.001_bs32_cjbn0.1_optadam_wd0.0001'
outd = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/Paolo_selected_t_396_corr/embeddings_test_sets'

resnet = 18
lr = 1e-4
for i in range(5):
    print(f"training split{i}")
    outdir = opj(*[outd, f'split{i}'])
    ckpth_best = opj(*[resd, f'split{i}', [x for x in os.listdir(opj(resd, f'split{i}')) if x.endswith('.ckpt')][0]])
    # print(ckpth_best)
    os.makedirs(outdir, exist_ok=True)
    train_dir = opj(d, f'split{i}_training')
    val_dir = opj(d, f'split{i}_validation')
    test_dir = opj(d, f'split{i}_test')
    cmd = f'CUDA_VISIBLE_DEVICES=0 python resnet_classifier_ml4time_get_embeddings_test_set.py {resnet} {nclasses} {epochs} {train_dir} {val_dir} -ts {test_dir} --test_only -g 1 -tr -s {outdir} --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805 -lr {lr} -tb_outdir {outdir} --ckpth_best {ckpth_best}'
    print(cmd)
    os.system(cmd)
    print('#'*50)
    # break