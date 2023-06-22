import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from grad_cam import grad_cam
import warnings
from argparse import ArgumentParser
from pathlib import Path
import os
# warnings.filterwarnings("ignore")

import torch
from torchvision import transforms
from resnet_classifier import ResNetClassifier
import numpy as np
def boxer_example():
    parser = ArgumentParser()
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
    parser.add_argument(
        "-o",
        "--optimizer",
        help="""PyTorch optimizer to use. Defaults to adam.""",
        default="adam",
    )
    parser.add_argument(
        "--img_dir",
        default=None,
    )
    parser.add_argument(
        "--modelckp",
        default=None,
    )
    parser.add_argument(
        "--outputdir",
        default=".",
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
    parser.add_argument('-cew','--ce_weights', nargs='+', type=float, default=None, required=False)

    args = parser.parse_args()

    
    os.makedirs(args.outputdir, exist_ok=True)
    if args.img_dir is not None:
        img_paths = [os.path.join(args.img_dir, x) for x in os.listdir(args.img_dir)]
    else:
        img_paths = ["/mnt/lsdf_iai-aida/Daten_Deininger/projects/2d_ms_org_ventricle_seg/data/dataset/val/ventricle/d16_B2A_12.jpg"]
    layer = {
            'layer1a': (1, 0),
            'layer1b': (1, 1),
            'layer1c': (1, 2),
            'layer2a': (2, 0),
            'layer2b': (2, 1),
            'layer2c': (2, 2),
            'layer2d': (2, 3),
            'layer3a': (3, 0),
            'layer3b': (3, 1),
            'layer3c': (3, 2),
            'layer3d': (3, 3),
            'layer3e': (3, 4),
            'layer3f': (3, 5),
            'layer4a': (4, 0),
            'layer4b': (4, 1),
            'layer4c': (4, 2)}
    for desc, layer_id in layer.items():
        print(desc)
        # define model
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
            ce_weights=args.ce_weights
            )
        model.load_state_dict(torch.load(args.modelckp)['state_dict'])
        model.eval()

        # get heatmap layer
        a, b = layer_id
        heatmap_layer = getattr(model.resnet_model, f'layer{a}')[b].conv2
        transform = transforms.Compose(
        [
            transforms.Resize((384, 512)),
            # transforms.RandomRotation(degrees=(0, 360), fill=255),
            # transforms.RandomResizedCrop(size=(384, 512), scale=(0.3, 1.0)),
            # transforms.ColorJitter(brightness=(0.9, 1.5), saturation=0.3, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
        # grad cam
        for img_path in img_paths:
            image = Image.open(img_path)
            input_tensor = transform(image)
            label = 0
            try:
                image_gradcam = grad_cam(model, input_tensor, heatmap_layer, label)
                plt.imshow(image_gradcam)
                plt.savefig(os.path.join(args.outputdir, f'{os.path.basename(img_path)}'.replace('.jpg', f'_gradcam_{desc}.jpg')))
                plt.close()
                plt.imshow(image)
                plt.savefig(os.path.join(args.outputdir, f'{os.path.basename(img_path)}'.replace('.jpg', '_orig.jpg')))
            except RuntimeError as e:
                pass
boxer_example()
