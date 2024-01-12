import os
from os.path import join as opj


slurm_backbone = '''#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=10000
#SBATCH --time=03:00:00
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
resnet = 50
d = '/hkfs/work/workspace_haic/scratch/oc9627-embryo_videos/videos_allt'
resd = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results_clip_durs_2_4'

for unif_temp_sub, batch_size in [(154, 2)]:
    for clip_dur in [2, 4]:
        for lr in [1e-3]:
            for i in range(5):
                # print(f"training split{i}")
                outdir_name = f'epochs{epochs}_resnet{resnet}_lr{lr}_cd{clip_dur}_uts{unif_temp_sub}_bs_{batch_size}'
                outdir = opj(*[resd, outdir_name, f'split{i}'])
                os.makedirs(outdir, exist_ok=True)
                train_dir = opj(d, f'split{i}_training')
                val_dir = opj(d, f'split{i}_validation')
                test_dir = opj(d, f'split{i}_test')
                cmd = f'python resnet_video_classifier.py {resnet}\\\n {nclasses}\\\n {epochs}\\\n {train_dir}\\\n {val_dir}\\\n --batch_size {batch_size}\\\n --unif_temp_sub {unif_temp_sub}\\\n --clip_duration {clip_dur}\\\n -ts {test_dir}\\\n -g 1 -tr -s {outdir}\\\n --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805\\\n -lr {lr}\\\n -tb_outdir {outdir}'
                print(slurm_backbone+cmd)
                with open(opj(slurm_scripts_dir, outdir_name+f'_split{i}.sh'), "w") as f:
                    f.write(slurm_backbone+cmd)
