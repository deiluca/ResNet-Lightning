import torch
import matplotlib
import warnings
from argparse import ArgumentParser
from pathlib import Path

warnings.filterwarnings("ignore")

# import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
# from torchmetrics import Accuracy, MatthewsCorrCoef
from torchvision import transforms
from torchvision.datasets import ImageFolder
# from pytorch_lightning.loggers import TensorBoardLogger

data_path = '/mnt/lsdf_iai-aida/Daten_Deininger/projects/2d_ms_org_ventricle_seg/data/dataset/train'
transform = transforms.Compose(
[
    # transforms.Resize((500, 500)),
    transforms.RandomRotation(degrees=(0, 360), fill=1),
    transforms.RandomResizedCrop(size=(768, 1024), scale=(0.75, 1.25)),
    transforms.ToTensor(),
    # transforms.Normalize((0.48232,), (0.23051,)),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
)

img_folder = ImageFolder(data_path, transform=transform)

dl = DataLoader(img_folder, batch_size=1, shuffle=False)
for img, label in dl:
    x = 1
    y = 1