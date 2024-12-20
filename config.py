from pathlib import Path
import random
import numpy as np
import torch
import json
import os
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

os.environ["NEURITE_BACKEND"] = 'pytorch'
torch.set_float32_matmul_precision('medium')
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_datapath():
    env_path = Path("env.json")
    if env_path.exists():
        with open(env_path) as f:
            env = json.load(f)
        return str(Path(env["datapath"]) / "data") + "/"
    else:
        return str(Path.cwd() / "data") + "/"


def set_seed(seed: int = 42):
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def arg():
    parser = ArgumentParser()

    parser.add_argument("--logger_name", type=str,
                        dest="logger_name",
                        default='default',
                        help="wandblogger")

    parser.add_argument("--network_type", type=str,
                        dest="network_type",
                        required=True,
                        help="relu or siren or finer")

    parser.add_argument("--dir_path", type=str,
                        dest="dir_path",
                        default='predicting_deformation_field',
                        help="predicting_pixel_intensities or predicting_deformation_field")

    parser.add_argument("--run_type", type=str,
                        dest="run_type",
                        default='training',
                        help="training/sanity_check/overfitting")

    parser.add_argument("--fast_dev_run",
                        type=bool,
                        dest="fast_dev_run",
                        default=False,
                        help="fast_dev_run")

    parser.add_argument("--resolution", type=int,
                        dest="resolution",
                        default='50',
                        help="number of generated pixel coordinate between pixel height/width, ideally should be greater than height and width")

    parser.add_argument("--similarity_metric", type=str,
                        dest="similarity_metric",
                        default='NCC',
                        help="MSE or NCC or MI")

    parser.add_argument("--regularization_type", type=str,
                        dest="regularization_type",
                        default='jacobian',
                        help="jacobian or bending_energy")

    parser.add_argument("--datapath", type=str,
                        default=get_datapath(),
                        help="path to data folder")

    parser.add_argument("--spatial_reg", type=float,
                        dest="spatial_reg",
                        default=0.01,
                        help="weight for spatial regularization")

    parser.add_argument("--temporal_reg", type=float,
                        dest="temporal_reg",
                        default=1.0,
                        help="weight for temporal regularization")

    parser.add_argument("--monotonicity_reg", type=float,
                        dest="monotonicity_reg",
                        default=0.5,
                        help="weight for monotonicity regularization")

    parser.add_argument("--subjectID", type=str,
                        dest="subjectID",
                        default="AD/005_S_0814",
                        help="subject to train, include patient type")

    parser.add_argument("--model_type", type=str,
                        dest="model_type",
                        default="pre",
                        help="pre/post/full"
                        )
    parser.add_argument("--loss_type", type=str,
                        dest="loss_type",
                        default="L2",
                        help="L1 or L2"
                        )

    parser.add_argument("--time", type=int,
                        dest="time",
                        nargs='+',
                        default=[0, 13, 14, 24],
                        help="time difference in months between scans 0 and each of t, start with 0")

    parser.add_argument("--batch_size", type=int,
                        dest="batch_size",
                        default=48, #for FD and 12 for AD
                        help="batch size")

    parser.add_argument("--patch_size", type=int,
                        dest="patch_size",
                        default=12,
                        help="patch size")

    parser.add_argument("--val_split", type=float,
                        dest="val_split",
                        default=0.3,
                        help="val split")

    parser.add_argument("--lr", type=float,
                        dest="lr",
                        default=1e-5,
                        help="learning rate")

    parser.add_argument("--weight_decay", type=float,
                        dest="weight_decay",
                        default=1e-5,
                        help="weight decay")

    parser.add_argument("--scale_factor", type=float,
                        dest="scale_factor",
                        # default=0.5,
                        default=1.0,
                        help="resolution of the image to to train with")

    parser.add_argument("--omega_0", type=float,
                        dest="omega_0",
                        default=30.0,
                        help="value of omega_0 for siren")

    parser.add_argument("--hidden_layers", type=int,
                        dest="hidden_layers",
                        default=5,
                        help="number of hidden layers")

    parser.add_argument("--hidden_features", type=int,
                        dest="hidden_features",
                        default=256,
                        help="number of hidden units per layer, this is scaled when concatenated with time features")

    parser.add_argument("--time_features", type=float,
                        dest="time_features",
                        default=64,
                        help="number of time features")

    parser.add_argument("--num_patches", type=int,
                        dest="num_patches",
                        default=3000,
                        help="total number of patches to be sampled for both train and val")

    parser.add_argument("--num_epochs", type=int,
                        dest="num_epochs",
                        default=150,
                        help="total number of epochs")

    parser.add_argument("--gradient_type", type=str,
                        dest="gradient_type",
                        default="analytic_gradient",
                        help="state to use either \"finite_difference\" or \"analytic_gradient\"")
    args = parser.parse_args()
    args.batch_size = 48 if args.gradient_type == "finite_difference" else 12
    #set LR and
    return args


def wandb_setup():
    args = arg()

    model_checkpoint = ModelCheckpoint(
        filename="{val_loss:.5f}-{epoch:02d}-" + args.subjectID.split("/")[-1] + "_" + args.logger_name,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
    )
    logger = WandbLogger(
        project="inrmorph",
        entity="aishalawal",
        name=args.logger_name,
        log_model=True,
    )
    # if not args.fast_dev_run:
    # logger.experiment.log_code()
    return model_checkpoint, logger


def save_logger_name(logger_name):
    with open('result/logger_name.txt', 'a') as f:
        f.write(logger_name)
        f.write('\n')
    f.close()
