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
from pytorch_lightning import seed_everything

from sklearn.metrics import auc

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
        weight_decay=0,
        save_path=None,
        cj_bn = 0.1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path
        self.cj_bn = cj_bn
        self.weight_decay = weight_decay

        self.optimizer = self.optimizers[optimizer]
        # instantiate loss criterion
        #if ce_weights is None:
        #    self.loss_fn = (
        #        nn.BCEWithLogitsLoss() if um_classes == 1 else nn.CrossEntropyLoss()
        #    )
        #else:
        #    self.loss_fn = (
        #        nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ce_weights[1])) if num_classes == 1 else nn.CrossEntropyLoss()
        #    )
        # create accuracy metric
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights))
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        #self.mcc = MatthewsCorrCoef(task='multiclass')
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

        #filenames, targets = [], []
        #for filename, target in ImageFolder(self.test_path).imgs:
        #    filenames.append(filename)
        #    targets.append(target)
        #self.df = pd.DataFrame.from_dict({'filename': filenames, 'target': targets})

        #self.test_predictions, self.test_targets = [], []

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

    def forward(self, X):
        return self.resnet_model(X)

    def configure_optimizers(self):
        print('using lr: ', self.lr)
        return self.optimizer(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _step(self, batch, mode):
        x, y = batch
        preds = self(x)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        #mcc = self.mcc(preds, y)
        if mode=='train':
            return loss, acc
        else:
            return loss, acc, preds.cpu().numpy(), y.cpu().numpy()

    def _dataloader(self, data_path, shuffle=False):
        # values here are specific to pneumonia dataset and should be updated for custom data
        transform = transforms.Compose(
            [
                # transforms.Resize((500, 500)),
                transforms.RandomRotation(degrees=(0, 360), fill=255),
                # transforms.RandomResizedCrop(size=(256, 256), scale=(0.3, 1.0), ratio=(1.0, 1.0)),
                transforms.ColorJitter(brightness=self.cj_bn, saturation=0.1, contrast=0.1),
                transforms.ToTensor(),
                # transforms.Normalize((0.48232,), (0.23051,)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img_folder = ImageFolder(data_path, transform=transform)
        print('img_folder.class_to_idx:', img_folder.class_to_idx)
        self.class_idx = img_folder.class_to_idx
        return DataLoader(img_folder, batch_size=self.batch_size, shuffle=shuffle)

    def train_dataloader(self):
        return self._dataloader(self.train_path, shuffle=True)

    def training_step(self, batch, batch_idx):
        loss, acc = self._step(batch, mode='train')
        # perform logging
        self.log(
            "Train/Loss", loss, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "Train/Acc", acc, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        return self._dataloader(self.val_path, shuffle=True)

    def validation_step(self, batch, batch_idx):
        loss, acc, _, _ = self._step(batch, mode='val')
        # perform logging
        self.log("Val/Loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("Val/Acc", acc, on_epoch=True, prog_bar=True, logger=True)

        # Calculate the confusion matrix using sklearn.metrics
        # cm = sklearn.metrics.confusion_matrix(targets, preds)
        
        # figure = plot_confusion_matrix(cm, class_names=['no_v', 'v'])
        # cm_image = plot_to_image(figure)
        # tensorboard = self.logger.experiment
        # tensorboard.add_image(cm_image)

    def test_dataloader(self):
        return self._dataloader(self.test_path)

    def test_step(self, batch, batch_idx):
        loss, acc, multacc, prauc, preds, targets = self._step(batch, mode='test')
        
        self.test_predictions.extend(np.squeeze(preds).tolist())
        self.test_targets.extend(np.squeeze(targets).tolist())
        print(len(self.test_predictions), self.df.shape[0])
        if len(self.test_predictions) == self.df.shape[0]:
            self.df['predicted'] = self.test_predictions
            self.df['target_2'] = self.test_targets
            self.df['predicted_cls'] = self.df['predicted'].apply(lambda x: np.argmax(x))
            print('saving output csv')
            self.df.to_csv(os.path.join(self.save_path, 'outputs.csv'), index=False)

        self.log("Test/Loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("Test/Acc", acc, on_epoch=True, prog_bar=True, logger=True)
        # self.log("Test/MCC", mcc, on_epoch=True, prog_bar=True, logger=True)
        
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
        "--ckpth_best",
        help="""for model testing: best model checkpoint""",
        type=str,
        default=None,
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
        "--test_only",
        help="do not train the model, only test model",
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
    parser.add_argument(
        "-wd",
        "--weight_decay",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=0.0,
    )
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f'{arg}: {value}')    
    seed_everything(42)

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
        save_path = args.save_path,
        cj_bn = args.cj_bn,
        weight_decay=args.weight_decay
    )

    save_path = args.save_path if args.save_path is not None else "./models"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_path,
        filename="resnet-model-{epoch}-{val_loss:.4f}",
        monitor="Val/Loss",
        save_top_k=1,
        mode="min",
        save_last=False,
    )

    stopping_callback = pl.callbacks.EarlyStopping(monitor='Val/Loss')

    # Instantiate lightning trainer and train model
    trainer_args = {
        "accelerator": "gpu" if args.gpus else None,
        "devices": [0],
        "strategy": "ddp",
        "max_epochs": args.num_epochs,
        "callbacks": [checkpoint_callback],
        "precision": 16 if args.mixed_precision else 32,
        "logger": TensorBoardLogger(args.tb_outdir, name="my_model"),
        "log_every_n_steps": 10**10,
        "deterministic": True,
        "auto_lr_find":args.use_autofindlr
    }
    trainer = pl.Trainer(**trainer_args)
    if not args.test_only:
        if args.use_autofindlr:
            # Run learning rate finder
            lr_finder = trainer.tuner.lr_find(model)
            # update hparams of the model
            model.lr = lr_finder.suggestion()
        trainer.fit(model)
        torch.save(trainer.model.resnet_model.state_dict(), save_path + "/trained_model.pt")

    if args.test_set:
        if not args.test_only:
            trainer.test(model, ckpt_path='best')
        else:
            assert args.ckpth_best is not None
            trainer.test(model, ckpt_path=args.ckpth_best)


