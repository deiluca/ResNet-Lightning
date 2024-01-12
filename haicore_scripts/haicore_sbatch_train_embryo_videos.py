import os
from os.path import join as opj

sbatch_base = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:full:1
#SBATCH --mem=30000
#SBATCH --time=3:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=luca.deininger@kit.edu
#SBATCH --job-name=Embryos
#SBATCH --output=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.out
#SBATCH --error=/home/iai/oc9627/ResNet-Lightning/logs/%x.%j.err
#SBATCH --partition=normal
#SBATCH --constraint="LSDF"
source ~/aidaretreat23/bin/activate\\\n\n"""

epochs = 100
model = 'mvit'
batch_size = 1
num_classes = 3
d = '/hkfs/work/workspace_haic/scratch/oc9627-embryo_videos/videos_allt/'
resd = '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results/'

clip_dur = 8
utf = 8
bs = 1
img_size = 224
wd = 1e-4
opt = 'sgd'
cj_bn = 0.3


for split in range(5):
    for lr in [1e-4, 1e-3]:
        outname = f'epochs{epochs}_{model}_lr-{lr}_cd{clip_dur}_uts8_bs1_img_size224_wd0.0001_opt-sgd_cj-bn0.5'
        sbatch_file = opj('sbatch_files', f'{outname}_split{split}.sh')
        outdir = opj(*[resd, outname, f'split{split}'])
        os.makedirs(outdir, exist_ok=True)
        traindir = opj(outdir, f'split{split}_training')
        valdir = opj(outdir, f'split{split}_validation')
        testdir = opj(outdir, f'split{split}_test')
        
        cmd = f"""python resnet_video_classifier.py {model}\
        {num_classes}\
        {epochs}\
        {traindir}\
        {valdir}\
        --batch_size {bs}\
        --unif_temp_sub {utf}\
        --clip_duration {clip_dur}\
        -ts {testdir}\
        -g 1\
        -tr\
        -s /lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results/epochs100_mvit_lr-autofind_cd8_uts8_bs1_img_size224_wd0.0001_opt-sgd_cj-bn0.5_config2/split0\
        --ce_weights 0.4722222222222222 2.0816326530612246 2.4878048780487805\
        -lr 0.001\
        -tb_outdir /lsdf/kit/iai/projects/iai-aida/Daten_Deininger/projects/embryo_project/training_datasets/videos_allt/results/epochs100_mvit_lr-autofind_cd8_uts8_bs1_img_size224_wd0.0001_opt-sgd_cj-bn0.5_config2/split0\
        --use_autofindlr \
        --vit_img_size 224\
        --weight_decay 0.0001\
        --optimizer sgd\
        --cj_bn 0.5"""