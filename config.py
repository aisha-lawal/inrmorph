from pathlib import Path
import random
import numpy as np
from typing import ClassVar
import torch
import json
import os
from enum import Enum
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger

os.environ["NEURITE_BACKEND"] = 'pytorch'
torch.set_float32_matmul_precision('medium')
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_datapath():
    env_path = Path("env.json")
    if env_path.exists():
        with open(env_path) as f:
            env = json.load(f)
        return str(Path(env["datapath"]) / "data") + "/"
    else:
        return str(Path.cwd() / "data") + "/"


# def set_seed(seed: int = 42):
#     torch.random.manual_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)


#define ENUMS
class NetworkType(Enum):
    FINER: ClassVar[str] = "finer"
    SIREN: ClassVar[str] = "siren"
    ReLU: ClassVar[str] = "relu"

class GradientType(Enum):
    FINITE_DIFFERENCE: ClassVar[str] = "finite_difference" #numerical approx
    ANALYTIC_GRADIENT: ClassVar[str] = "analytic_gradient"

class SimilarityMetric(Enum): #using just NCC for now
    NCC: ClassVar[str] = "NCC" 
    MSE: ClassVar[str] = "MSE"

class SpatialRegularizationType(Enum):
    SPATIAL_RATE_OF_TEMPORAL_CHANGE: ClassVar[str] = "spatial_rate_of_temporal_change" 
    SPATIAL_JACOBIAN_MATRIX_PENALTY: ClassVar[str] = "spatial_jacobaian_matrix_penalty"


def arg():
    parser = ArgumentParser()

    parser.add_argument("--logger_name", type=str,
                        dest="logger_name",
                        default='default',
                        help="wandblogger")

    parser.add_argument("--network_type",
                        type=lambda s: NetworkType(s),
                        choices=list(NetworkType),
                        # required=True,
                        default=NetworkType.SIREN,
                        help="Choose an option from the Enum: {}".format(", ".join(e.name for e in NetworkType)))


    parser.add_argument("--dir_path", type=str,
                        dest="dir_path",
                        default='predicting_deformation_field',
                        help="predicting_pixel_intensities or predicting_deformation_field")

    parser.add_argument("--run_type", type=str,
                        dest="run_type",
                        default='training',
                        help="training/sanity_check/overfitting")

    parser.add_argument("--fast_dev_run",
                        type=bool, #with arg parse passing a value is equivalent to passing True
                        dest="fast_dev_run",
                        default=False,
                        help="fast_dev_run")

    parser.add_argument("--extrapolate",
                        type=bool,
                        dest="extrapolate",
                        default=False,
                        help="extrapolate beyond observed timepoint")

    parser.add_argument("--fixed_time_embedding",
                        type=bool,
                        dest="fixed_time_embedding",
                        default=False,
                        help="to use fixed time embedding or generate embedding on the fly")

    parser.add_argument("--interpolate",
                        type=bool,
                        dest="interpolate",
                        default=False,
                        help="interpolate between observed timepoint")

    parser.add_argument("--add_noise",
                        type=bool,
                        dest="add_noise",
                        default=False,
                        help="add gaussian noise to follow up scans")

    parser.add_argument("--resolution", type=int,
                        dest="resolution",
                        default='50',
                        help="number of generated pixel coordinate between pixel height/width, ideally should be greater than height and width")

    parser.add_argument("--similarity_metric",
                        type=lambda s: SimilarityMetric(s),
                        choices=list(SimilarityMetric),
                        default=SimilarityMetric.NCC,
                        help="Choose an option from the Enum: {}".format(", ".join(e.name for e in SimilarityMetric)))

    parser.add_argument("--datapath", type=str,
                        default=get_datapath(),
                        help="path to data folder")

    parser.add_argument("--spatial_reg", type=float,
                        dest="spatial_reg",
                        # default=0.01, #current default
                        default=0.1,
                        help="weight for spatial regularization")

    parser.add_argument("--l2_weight", type=float,
                        dest="l2_weight",
                        default=100,
                        # default=10,
                        help="l2 regularization weight")

    parser.add_argument("--temporal_reg", type=float,
                        dest="temporal_reg",
                        # default=1.0,
                        default=0.1,

                        help="weight for temporal regularization")

    parser.add_argument("--monotonicity_reg", type=float,
                        dest="monotonicity_reg",
                        # default=0.5,
                        default=0.1,
                        help="weight for monotonicity regularization")

    parser.add_argument("--subjectID", type=str,
                        dest="subjectID",
                        # default="MCI/002_S_1155",
                        default="AD/005_S_0814",
                        required=False,
                        help="subject to train, include patient type")
    
    parser.add_argument("--spatial_reg_type",
                        type=lambda s: SpatialRegularizationType(s),
                        choices=list(SpatialRegularizationType),
                        default=SpatialRegularizationType.SPATIAL_JACOBIAN_MATRIX_PENALTY,
                        # default=SpatialRegularizationType.SPATIAL_RATE_OF_TEMPORAL_CHANGE,
                        help="Choose an option from the Enum: {}".format(", ".join(e.name for e in SpatialRegularizationType)))

    parser.add_argument("--batch_size", type=int,
                        dest="batch_size",
                        default=48, #for FD and 12 for AD
                        help="batch size")

    parser.add_argument("--patch_size", type=int,
                        dest="patch_size",
                        default=12,
                        # default=16,
                        help="patch size")

    parser.add_argument("--val_split", type=float,
                        dest="val_split",
                        default=0.3,
                        help="val split")

    parser.add_argument("--lr", type=float,
                        dest="lr",
                        # default=1e-5,
                        default=4e-5,
                        help="learning rate")

    parser.add_argument("--weight_decay", type=float,
                        dest="weight_decay",
                        default=4e-5,
                        help="weight decay")

    parser.add_argument("--scale_factor", type=float,
                        dest="scale_factor",
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

    parser.add_argument("--time_features", type=int,
                        dest="time_features",
                        default=64,
                        help="number of time features")

    parser.add_argument("--num_patches", type=int,
                        dest="num_patches",
                        default=2000,
                        # default=3000,
                        help="total number of patches to be sampled for both train and val")

    parser.add_argument("--num_epochs", type=int,
                        dest="num_epochs",
                        default=90,
                        # default=150,
                        help="total number of epochs")

    parser.add_argument("--noise_std", type=float,
                        dest="noise_std",
                        default=0.5,
                        help="standard deviation of gaussian noise")

    parser.add_argument("--noise_mean", type=float,
                        dest="noise_mean",
                        default=0.0,
                        help="mean of gaussian noise")

    parser.add_argument("--gradient_type",
                        type=lambda s: GradientType(s),
                        choices=list(GradientType),
                        default=GradientType.ANALYTIC_GRADIENT,
                        help="Choose an option from the Enum: {}".format(", ".join(e.name for e in GradientType)))

    args = parser.parse_args()
    args.batch_size = 48 if args.gradient_type == "finite_difference" else 12
    # args.batch_size = 48 if args.gradient_type == "finite_difference" else 8
    print(f"args: {args.extrapolate, args.add_noise}")
    return args


def wandb_setup():
    args = arg()

    model_checkpoint = ModelCheckpoint(
        filename="{val_loss:.5f}-{epoch:02d}-" + "_" + args.logger_name,
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
    logger.experiment.log_code()
    return model_checkpoint, logger


def save_logger_name(logger_name):
    with open('result/logger_name.txt', 'a') as f:
        f.write(logger_name)
        f.write('\n')
    f.close()


#delay checkpointing until after a certain epoch
class DelayedModelCheckpoint(ModelCheckpoint):
    def __init__(self, start_epoch, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch >= self.start_epoch:
            super().on_validation_end(trainer, pl_module)
