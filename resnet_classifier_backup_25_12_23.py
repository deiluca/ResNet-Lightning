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
from torchmetrics import Accuracy, MatthewsCorrCoef
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pytorch_lightning.loggers import TensorBoardLogger

import sklearn
from utils import plot_confusion_matrix, plot_to_image

import datetime
import pandas as pd
import numpy as np
# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "sgd": SGD}

    def __init__(
        self,
        num_classes,
        resnet_version,
        train_path,
        val_path,
        test_path=None,
        optimizer="adam",
        lr=1e-3,
        batch_size=16,
        transfer=True,
        tune_fc_only=True,
        ce_weights = None,
        save_path=None
    ):
        super().__init__()

        self.num_classes = num_classes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        if ce_weights is None:
            self.loss_fn = (
                nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
            )
        else:
            self.loss_fn = (
                nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ce_weights[1])) if num_classes == 1 else nn.CrossEntropyLoss(weight=torch.tensor(ce_weights))
            )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        self.mcc = MatthewsCorrCoef(task='binary')
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)

        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        # datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        # self.test_dir = f'model_test_predictions/{datestring}'
        # os.makedirs(self.test_dir, exist_ok=True)

        filenames, targets = [], []
        for filename, target in ImageFolder(self.test_path).imgs:
            filenames.append(filename)
            targets.append(target)
        self.df = pd.DataFrame.from_dict({'filename': filenames, 'target': targets})

        self.test_predictions, self.test_targets = [], []

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, batch, mode):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        mcc = self.mcc(preds, y)
        if mode=='train':
            return loss, acc, mcc
        else:
            return loss, acc, mcc, preds.cpu().numpy(), y.cpu().numpy()

    def _dataloader(self, data_path, shuffle=False):
        # values here are specific to pneumonia dataset and should be updated for custom data
        transform = transforms.Compose(
            [
                # transforms.Resize((500, 500)),
                transforms.RandomRotation(degrees=(0, 360), fill=255),
                transforms.RandomResizedCrop(size=(384, 512), scale=(0.3, 1.0), ratio=(1.0, 1.0)),
                transforms.ColorJitter(brightness=0.1, saturation=0.1, contrast=0.1),
                transforms.ToTensor(),
                # transforms.Normalize((0.48232,), (0.23051,)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img_folder = ImageFolder(data_path, transform=transform)

        return DataLoader(img_folder, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self._dataloader(self.train_path, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss, acc, mcc = self._step(batch, mode='train')
        # perform logging
        self.log(
            "Train/Loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train/Acc", acc, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train/MCC", mcc, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        return self._dataloader(self.val_path, shuffle=True)

    def validation_step(self, batch, batch_idx):
        loss, acc, mcc, _, _ = self._step(batch, mode='val')
        # perform logging
        self.log("Val/Loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("Val/Acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("Val/MCC", mcc, on_epoch=True, prog_bar=True, logger=True)

        # Calculate the confusion matrix using sklearn.metrics
        # cm = sklearn.metrics.confusion_matrix(targets, preds)
        
        # figure = plot_confusion_matrix(cm, class_names=['no_v', 'v'])
        # cm_image = plot_to_image(figure)
        # tensorboard = self.logger.experiment
        # tensorboard.add_image(cm_image)

    def test_dataloader(self):
        return self._dataloader(self.test_path)

    def test_step(self, batch, batch_idx):
        loss, acc, mcc, preds, targets = self._step(batch, mode='test')
        
        self.test_predictions.extend(np.squeeze(preds).tolist())
        self.test_targets.extend(np.squeeze(targets).tolist())
        print(len(self.test_predictions), self.df.shape[0])
        if len(self.test_predictions) == self.df.shape[0]:
            self.df['predicted'] = self.test_predictions
            self.df['target_2'] = self.test_targets
            self.df.to_csv(os.path.join(self.save_path, 'outputs.csv'), index=False)

        self.log("Test/Loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("Test/Acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("Test/MCC", mcc, on_epoch=True, prog_bar=True, logger=True)
        
        # perform logging
        # self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        # self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)


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

    # # Instantiate Model
    model = ResNetClassifier(
        num_classes=args.num_classes,
        resnet_version=args.model,
        train_path=args.train_set,
        val_path=args.val_set,
        test_path=args.test_set,
        optimizer=args.optimizer,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        transfer=args.transfer,
        tune_fc_only=args.tune_fc_only,
        ce_weights=args.ce_weights,
        save_path = args.save_path
    )

    save_path = args.save_path if args.save_path is not None else "./models"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
        monitor="Val/Loss",
        save_top_k=3,
        mode="min",
        save_last=False,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor='Val/Loss')

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

    trainer.fit(model)

    if args.test_set:
        trainer.test(model)
    # Save trained model weights
    torch.save(trainer.model.resnet_model.state_dict(), save_path + "/trained_model.pt")
