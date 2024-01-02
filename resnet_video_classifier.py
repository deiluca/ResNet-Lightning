import os
import warnings
from argparse import ArgumentParser
from pathlib import Path

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MatthewsCorrCoef, PrecisionRecallCurve
# from torcheval.metrics import MulticlassAUPRC, MulticlassAccuracy, 
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import TensorBoardLogger

import sklearn
from utils import plot_confusion_matrix, plot_to_image

import datetime
import pandas as pd
import numpy as np

from sklearn.metrics import auc

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorchvideo.models.resnet
import pytorchvideo
from pytorchvideo.data import Kinetics

from collections import Counter
from os.path import join as opj


def make_kinetics_resnet():
    return 

class KineticsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_path,
        val_path,
        batch_size,
        clip_dur,
        unif_temp_sub=16,
        test_path=None,
        num_workers=8,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.unif_temp_sub = unif_temp_sub

        self.clip_dur = clip_dur

    def _dataloader(self, data_path, mode):
        # values here are specific to pneumonia dataset and should be updated for custom data
        transform = Compose(
            [
            ApplyTransformToKey(
              key="video",
              transform=Compose(
                  [
                    UniformTemporalSubsample(self.unif_temp_sub),### subsamples 8 images from T in range 0-153, so it will select at the beginning, middle and end, so probably day 0, 20, 40, 60, 80, 100, 120, 140 (these are 8)
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    # RandomShortSideScale(min_size=256, max_size=320),
                    # RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                  ]
                ),
              ),
            ]
        )
        dataset = Kinetics(
            data_path=os.path.join(data_path),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform" if mode!='train' else 'random', self.clip_dur), #original : random, what is this?
            decode_audio=False,
            transform=transform
        )
        
        # labels = []
        # for elem in dataset:
        #     labels.append(elem['label'])

        # print("Class Frequencies:", Counter(labels))
        # print(dataset.classes)
        # torch.unique(dataset.targets, return_counts=True)
        print(mode, "dataset.num_videos:", dataset.num_videos)
        return torch.utils.data.DataLoader(dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers
        )   

    def train_dataloader(self):
        return self._dataloader(self.train_path, mode='train')
    
    def val_dataloader(self):
        return self._dataloader(self.val_path, mode='val')
    
    def test_dataloader(self):
        return self._dataloader(self.test_path, mode='test')
    
class VideoClassificationLightningModule(pl.LightningModule):
    def __init__(self,
                 resnet_version,
                 num_classes,
                 ce_weights,
                 test_path,
                 save_path,
                 lr=1e-1):
        super().__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.test_path = test_path
        self.save_path = save_path

        self.model = pytorchvideo.models.resnet.create_resnet(
                    input_channel=3, # RGB input from Kinetics
                    model_depth=resnet_version, # For the tutorial let's just use a 50 layer network
                    model_num_class=num_classes, # Kinetics has 400 classes so we need out final head to align
                    norm=nn.BatchNorm3d,
                    activation=nn.ReLU
        )
        self.loss_fn = (
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ce_weights[1])) if self.num_classes == 1 else nn.CrossEntropyLoss(weight=torch.tensor(ce_weights))
            )
                # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=self.num_classes
        )
        self.prcurve = PrecisionRecallCurve(task='multiclass', num_classes=self.num_classes, average=None)

        self.multacc = Accuracy(task='multiclass', num_classes=num_classes, average=None)
        self.class_idx = {}
        self.class_idx['bad'] = 0
        self.class_idx['good'] = 1
        self.class_idx['mediocre'] = 2

        self.df = self.get_test_gt()

        self.test_predictions, self.test_targets = [], []

    def get_test_gt(self):
        filenames, targets = [], []

        # class_to_idx = {'bad':0,
        #                 'good': 1,
        #                 'mediocre':2}

        for subd in ['bad', 'good', 'mediocre']:
            d2 = opj(self.test_path, subd)
            for x in os.listdir(d2):
                filenames.append(x)
                targets.append(self.class_idx[subd])
        df = pd.DataFrame.from_dict({'filename': filenames, 'target': targets})
        return df
    
    def forward(self, x):
        return self.model(x)
    
    def get_prauc(self, prcurve):
                # Calculate AUC for each class
        class_auc_list = []
        for precision, recall, thresholds in zip(prcurve[0], prcurve[1], prcurve[2]):
            mask = ~torch.isnan(recall)
            # Check if there are enough points to compute AUC
            if len(mask.nonzero()) >= 2:
                class_auc = auc(recall[mask].cpu().numpy(), precision[mask].cpu().numpy())
                class_auc_list.append(class_auc)
            else:
                # print("Not enough points to compute AUC for this class.")
                class_auc_list.append(np.nan)

        return class_auc_list
    
    def _step(self, batch, mode):
        x = batch["video"]
        y = batch["label"]
        preds = self.model(x)

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        multacc = self.multacc(preds, y)
        prcurve = self.prcurve(preds, y)
        prauc = self.get_prauc(prcurve)
        self.log(
            f"{mode}/Loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        mode_lower = mode.lower()
        self.log(
            f"{mode_lower}_loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"{mode}/Acc", acc, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"{mode}/AccBad", multacc[self.class_idx['bad']], on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"{mode}/AccMediocre", multacc[self.class_idx['mediocre']], on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"{mode}/AccGood", multacc[self.class_idx['good']], on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"{mode}/PRAUCBad", prauc[self.class_idx['bad']], on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"{mode}/PRAUCMediocre", prauc[self.class_idx['mediocre']], on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"{mode}/PRAUCGood", prauc[self.class_idx['good']], on_epoch=True, prog_bar=False, logger=True
        )
        if mode=='Test':
            return loss, preds.cpu().numpy(), y.cpu().numpy()
        else:
            return loss


    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        loss = self._step(batch, mode='Train')

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, mode='Val')

        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._step(batch, mode='Test')
        
        self.test_predictions.extend(np.squeeze(preds).tolist())
        self.test_targets.extend(np.squeeze(targets).tolist())
        print(len(self.test_predictions), self.df.shape[0])
        if len(self.test_predictions) == self.df.shape[0]:
            self.df['predicted'] = self.test_predictions
            self.df['target_2'] = self.test_targets
            self.df['predicted_cls'] = self.df['predicted'].apply(lambda x: np.argmax(x))
            print('saving output csv')
            self.df.to_csv(os.path.join(self.save_path, 'outputs.csv'), index=False)


    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "model",
        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
        type=int,
    )
    parser.add_argument(
        "num_classes", help="""Number of classes to be learned.""", type=int
    )
    parser.add_argument("num_epochs", help="""Number of Epochs to Run.""", type=int)
    parser.add_argument(
        "train_set", help="""Path to training data folder.""", type=Path
    )
    parser.add_argument("val_set", help="""Path to validation set folder.""", type=Path)
    # Optional arguments
    parser.add_argument(
        "-amp",
        "--mixed_precision",
        help="""Use mixed precision during training. Defaults to False.""",
        action="store_true",
    )
    parser.add_argument(
        "-ts", "--test_set", help="""Optional test set path.""", type=Path
    )
    parser.add_argument('-cew','--ce_weights', nargs='+', type=float, default=None, required=False)
    parser.add_argument(
        "-o",
        "--optimizer",
        help="""PyTorch optimizer to use. Defaults to adam.""",
        default="adam",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-cd",
        "--clip_duration",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-uts",
        "--unif_temp_sub",
        type=int,
        default=8,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="""Manually determine batch size. Defaults to 16.""",
        type=int,
        default=16,
    )
    parser.add_argument(
        "-tr",
        "--transfer",
        help="""Determine whether to use pretrained model or train from scratch. Defaults to True.""",
        action="store_true",
    )
    parser.add_argument(
        "-to",
        "--tune_fc_only",
        help="Tune only the final, fully connected layers.",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--save_path", help="""Path to save model trained model checkpoint."""
    )
    parser.add_argument(
        "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None
    )
    parser.add_argument(
        "-tb_outdir", "--tb_outdir", help="""tb_outdir""", type=Path
    )
    args = parser.parse_args()    
    model = VideoClassificationLightningModule(resnet_version=args.model,
                                               num_classes=args.num_classes,
                                               ce_weights=args.ce_weights,
                                               test_path = args.test_set,
                                               lr=args.learning_rate,
                                               save_path=args.save_path)
    data_module = KineticsDataModule(train_path = args.train_set,                  
                                    val_path = args.val_set,
                                    test_path = args.test_set,
                                    clip_dur=args.clip_duration,
                                    unif_temp_sub = args.unif_temp_sub,
                                    batch_size=args.batch_size,
)

    save_path = args.save_path if args.save_path is not None else "./models"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor='val_loss')

    # Instantiate lightning trainer and train model
    trainer_args = {
        "accelerator": "gpu" if args.gpus else None,
        "devices": [0],
        # "strategy": "dp" if args.gpus > 1 else None,
        "strategy": "ddp" if args.gpus > 1 else None,
        "max_epochs": args.num_epochs,
        "callbacks": [checkpoint_callback],
        "precision": 16 if args.mixed_precision else 32,
        "logger": TensorBoardLogger(args.tb_outdir, name="my_model"),
        "log_every_n_steps": 10**10
    }
    trainer = pl.Trainer(**trainer_args)

    trainer.fit(model, data_module)

    if args.test_set:
        trainer.test(model, data_module)
