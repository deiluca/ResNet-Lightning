import os
from os.path import join as opj


slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=30000
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
source ~/aidaretreat23/bin/activate\n\n'''

slurm_scripts_dir = "sbatch_files"

epochs = 100
nclasses = 3

d = '/hkfs/work/workspace_haic/scratch/oc9627-embryo_videos/videos_allt'
resd = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results'

weight_decay = 1e-4
opt = 'sgd'
for unif_temp_sub, vit_img_size, batch_size in [(8, 224, 1)]:
    for clip_dur in [8]:
        for lr in [1e-3]:
            for i in range(5):
                # print(f"training split{i}")
                outdir_name = f'epochs{epochs}_mvit_lr-autofind_cd{clip_dur}_uts{unif_temp_sub}_bs{batch_size}_img_size{vit_img_size}_wd{weight_decay}_opt-{opt}_config2'
                outdir = opj(*[resd, outdir_name, f'split{i}'])
                ckpth_best = opj(outdir, [x for x in os.listdir(outdir) if x.endswith('.ckpt')][0])
                train_dir = opj(d, f'split{i}_training')
                val_dir = opj(d, f'split{i}_validation')
                test_dir = opj(d, f'split{i}_test')
                cmd = f'python resnet_video_classifier.py mvit\\\n {nclasses}\\\n {epochs}\\\n {train_dir}\\\n {val_dir}\\\n --batch_size {batch_size}\\\n --unif_temp_sub {unif_temp_sub}\\\n --clip_duration {clip_dur}\\\n -ts {test_dir}\\\n -g 1 -tr -s {outdir}\\\n --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805\\\n -lr {lr}\\\n -tb_outdir {outdir}\\\n --use_autofindlr \\\n --vit_img_size {vit_img_size}\\\n --weight_decay {weight_decay}\\\n --optimizer {opt}\\\n --test_only\\\n --ckpth_best {ckpth_best}'

                with open(opj(slurm_scripts_dir, outdir_name+f'_split{i}.sh'), "w") as f:
                    f.write(slurm_backbone+cmd)
