import torch
import nibabel as nib
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Any
import glob
import numpy as np
import os
from torch.nn import functional as F
from argparse import ArgumentParser
import lightning.pytorch as pl
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import matplotlib.pyplot as plt
import matplotlib
import torch.nn.init as init
import monai
import wandb
import csv
os.environ["NEURITE_BACKEND"] = 'pytorch'
import neurite as ne
# matplotlib.use('Agg')
# torch.manual_seed(42)

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = ('cuda:0' if torch.cuda.is_available() else 'cpu')




def arg():
    parser = ArgumentParser()

    parser.add_argument("--logger_name", type=str,
                dest="logger_name",
                default='default',
                help="wandblogger")
    
    parser.add_argument("--model", type=str,
                dest="model",
                default='hyperreg',
                help="sirenreg or hyperreg")
    
    parser.add_argument("--network_type", type=str,
                dest="network_type",
                default='siren',
                help="relu or siren")
    
    parser.add_argument("--dir_path", type=str,
                dest="dir_path",
                default='predicting_deformation_field',
                help="predicting_pixel_intensities or predicting_deformation_field")
    
    parser.add_argument("--run_type", type=str,
                dest="run_type",
                default='training',
                help="training/sanity_check/overfitting")
    
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
       
    parser.add_argument("--spatial_reg", type=float,
                dest="spatial_reg",
                default=0.1,
                help="weight for spatial regulatization")
    
    parser.add_argument("--temporal_reg", type=float,
                dest="temporal_reg",
                default=0.1,
                help="weight for temporal regulatization")
    
    parser.add_argument("--subjectID", type=str,
                dest="subjectID",
                default="ad/005_S_0814",
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
                # default=[0, 13],
                help="time difference in months between scans 0 and each of t, start with 0")

    parser.add_argument("--gradient_type", type=str,
                        dest="gradient_type",
                        default="finite_difference",
                        help="state to use either \"finite_difference\" or \"direct_gradient\"")
    args = parser.parse_args()
    return args


def wandb_setup():
    args = arg()

    model_checkpoint = ModelCheckpoint(
        filename= "{val_loss:.5f}-{epoch:02d}-" + args.subjectID.split("/")[-1] +"_"+ args.logger_name,
        save_top_k = 1,
        monitor="val_loss",
        mode = "min",
    )

    early_stopping = EarlyStopping(
        monitor= "val_loss",
        patience= 8,
        mode= "min",
    )
    logger = WandbLogger(
        project= "lowhighfieldreg",
        # name = args.network_type + "_" + args.logger_name,
        name =args.logger_name,
        log_model= "all",
    )
    # logger.experiment.log_code(".")

    return model_checkpoint, early_stopping, logger


def save_artifacts(trainer, model_checkpoint,logger_name, logger):

    #creates new artifact and saves the last model not best
    # artifact_name = model_checkpoint.filename.format(
    # val_loss=trainer.callback_metrics["val_loss"].item(),
    # epoch=trainer.current_epoch
    # ) 
    # artifact = wandb.Artifact(
    #     name=logger_name,
    #     # name=artifact_name,
    #     type="model",
    #     description="low high field reg"
    # )
    # artifact.add_file(model_checkpoint.best_model_path, name=artifact_name+".ckpt")

    # logger.experiment.log_artifact(artifact)
    with open('logger_name.txt', 'a') as f:
        f.write(logger_name)
        f.write('\n')
    f.close()