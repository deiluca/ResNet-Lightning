import os
from os.path import join as opj

epochs = 100
nclasses = 3
resnet = 50

resd = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results'
d = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt'

for unif_temp_sub, batch_size in [(16, 16), (32, 8), (64, 4), (154, 1)]:
    for clip_dur in [8]:
        for lr in [1e-5, 1e-6, 1e-4, 1e-3, 1e-2]:
            for i in range(5):
                print(f"training split{i}")
                outdir = opj(*[resd, f'epochs{epochs}_resnet{resnet}_lr{lr}_cd{clip_dur}_uts{unif_temp_sub}_bs_{batch_size}', f'split{i}'])
                os.makedirs(outdir, exist_ok=True)
                train_dir = opj(d, f'split{i}_training')
                val_dir = opj(d, f'split{i}_validation')
                test_dir = opj(d, f'split{i}_test')
                cmd = f'CUDA_VISIBLE_DEVICES=0 python resnet_classifier_video.py {resnet} {nclasses} {epochs} {train_dir} {val_dir} --batch_size {batch_size} --unif_temp_sub {unif_temp_sub} --clip_duration {clip_dur} -ts {test_dir} -g 1 -tr -s {outdir} --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805 -lr {lr} -tb_outdir {outdir}'
                
                

                
                break
                # os.system(cmd)
                print('#'*70)