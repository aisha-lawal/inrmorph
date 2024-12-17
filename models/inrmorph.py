from typing import Any, List
import torch.nn as nn
from torch import Tensor
from config import set_seed, device
import numpy as np
import torch
from models.finer import Finer
from models.relu import ReLU
from models.siren import Siren
from utils import SpatialTransform, SmoothDeformationField, MonotonicConstraint
import lightning as pl

class InrMorph(pl.LightningModule):
    def __init__(self,
                 I0: Tensor,
                 It: List,
                 patch_size: List,
                 spatial_reg_weight: float,
                 temporal_reg_weight: float,
                 monotonicity_reg_weight: float,
                 batch_size: int,
                 network_type: str,
                 similarity_metric: str,
                 gradient_type: str,
                 loss_type: str,
                 time: Tensor,
                 lr: float,
                 weight_decay: float,
                 omega_0: float,
                 hidden_layers: int,
                 time_features: int,
                 hidden_features: int,
                 ) -> None:
        super().__init__()
        self.I0 = I0
        self.It = It
        self.patch_size = patch_size
        self.spatial_reg_weight = spatial_reg_weight
        self.temporal_reg_weight = temporal_reg_weight
        self.monotonicity_reg_weight = monotonicity_reg_weight
        self.batch_size = batch_size
        self.network_type = network_type
        self.similarity_metric = similarity_metric
        self.gradient_type = gradient_type
        self.loss_type = loss_type
        self.time = time
        self.lr = lr
        self.weight_decay = weight_decay

        self.ndims = len(self.patch_size)
        self.nsamples = len(self.It)
        self.time_features = time_features
        self.in_features = self.ndims
        self.out_features = self.ndims
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = omega_0
        self.seed = 42
        # self.time = self.time.clone().detach().requires_grad_(True)
        self.time = self.time.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, self.batch_size,
                        *self.patch_size, -1).clone().detach().requires_grad_(True)
        self.first_omega = 30
        self.hidden_omega = 30
        self.init_method = "sine"
        self.init_gain = 1
        self.fbs = 5  # k for bias initialization according to paper optimal
        self.set_seed = set_seed(self.seed)
        assert self.loss_type in ["L1", "L2"], "Invalid loss type"
        assert self.gradient_type in ["finite_difference", "analytic_gradient"], "Invalid computation type"
        assert self.network_type in ["siren", "relu", "finer"], "Invalid network type"
        self.transform = SpatialTransform()
        self.t_mapping = self.time_mapping(self.time_features)
        self.smoothness = SmoothDeformationField(self.gradient_type, self.patch_size,
                                                 self.batch_size)

        self.monotonic_constraint = MonotonicConstraint(self.patch_size, self.batch_size, self.time)

        if self.network_type == "finer":

            self.mapping = Finer(in_features=self.in_features, out_features=self.out_features,
                                 hidden_layers=self.hidden_layers, hidden_features=self.hidden_features,
                                 first_omega=self.first_omega, hidden_omega=self.hidden_omega,
                                 init_method=self.init_method, init_gain=self.init_gain, fbs=self.fbs)

        elif self.network_type == "siren":

            self.mapping = Siren(layers=[self.in_features, *[self.hidden_features + (self.time_features * i) for i in
                                                             range(self.hidden_layers)], self.out_features],
                                 omega_0=self.omega_0, time_features=self.time_features)

        else:
            self.mapping = ReLU(layers=[self.in_features, *[self.hidden_features + i * (self.time_features * 2) for
                                                            i in range(self.num_layers)], self.out_features],
                                time_features=self.time_features)

    def forward(self, coords):

        displacement_t = []  # len samples
        for time in self.time:
            t_n = time.view(self.batch_size, coords.shape[1], 1)
            t_n = self.t_mapping(t_n)  # concat this with every layer.
            phi_dims = self.mapping(coords, t_n)  # single layer
            displacement_t.append(phi_dims)

        return displacement_t

    def training_step(self, batch, batch_idx):
        coords = batch
        train_loss = self.train_val_pipeline(coords, "train")

        return train_loss

    def validation_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        coords = batch
        val_loss = self.train_val_pipeline(coords, "val")
        return val_loss

    def train_val_pipeline(self, coords: torch.Tensor, process: str):
        # flatten to shape [batch_size, flattened_patch_size, ndims]
        coords = coords.view(self.batch_size, np.prod(self.patch_size), self.ndims)
        coords = coords.clone().detach().requires_grad_(True)
        displacement_t = self.forward(coords)
        warped_t, fixed, deformation_field_t = self.compute_transform(coords, displacement_t)

        ncc, spatial_smoothness, loss, mono_loss, temporal_smoothness = self.compute_loss(warped_t,
                                                                                          fixed, deformation_field_t, coords)

        metrics = {
            f"{process}_ncc": ncc,
            f"{process}_spatial_smoothness": spatial_smoothness,
            f"{process}_loss": loss,
            f"{process}_mono_loss": mono_loss,
            f"{process}_temporal_smoothness": temporal_smoothness,
        }

        for name, value in metrics.items():
            self.log(name, value, on_epoch=True, on_step=False)

        return loss

    def test_step(self, coords, time):
        coords = coords.unsqueeze(0)
        batch_size = 1
        tm = time.unsqueeze(0).unsqueeze(0)
        tm = tm.expand(batch_size, coords.shape[1], 1)

        tm = self.t_mapping(tm)
        displacement = self.mapping(coords, tm)

        return displacement

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return optimizer

    def compute_transform(self, coords, displacement):
        """
        Args:
        Here the baseline cooridnate (coords) serve as a reference coordinate which we align to. 
            - displacement (list of torch.tensor (*patch_size, ndims)): predicted displacement for each t in It, how  each point
            in the baseline should move to match the morphology of the followup at each point t
            - coords (torch.tensor): original coordinate used for warped
            - we warp the follow up image to the base line coordinate for each t, to have a uniform coordinate system (baseline)
            warp function at time = 0 should be the identity transform

        """
        deformation_field_t = []
        warped_t = []
        fixed = self.transform.trilinear_interpolation(coords=coords, img=self.I0).view(self.batch_size,
                                                                                        *self.patch_size)  # to reshape as a patch

        for idx, t in enumerate(self.time.unique()):
            deformation_field_t.append(torch.add(displacement[idx],
                                                 coords))  # apply the displacement relative to the baseline coordinate (coords)
            if idx != 0:
                warped_t.append(
                    self.transform.trilinear_interpolation(coords=deformation_field_t[idx], img=self.It[idx]).view(
                        self.batch_size, *self.patch_size))
        return warped_t, fixed, deformation_field_t

    def compute_loss(self, warped_t, fixed, deformation_field_t, coords):
        """
        Args: 
            warped_t (list of torch.Tensor): contains warped images at time t of shape
            [batch_size, *patch_size] i.e [batch_size, 32, 32, 32]

            fixed_t (list of torch.Tensor): list contains It at time t warped with original coords of shape 
            [batch_size, *patch_size] i.e [batch_size, 32, 32, 32]

            deformation_field_t (list of torch.Tensor): contains deformation field at time t of shape [batch_size, flattened_patch, ndims]

            coords: coordinats of shape [batch_size, flattened_patch, ndims]

        Returns: 

        """
        mono_loss = 0
        total_loss = 0
        similarity = 0
        spatial_smoothness = 0
        jac_det = torch.zeros(len(self.time.unique()), self.batch_size, np.prod(self.patch_size)).to(device)
        for idx, _ in enumerate(self.time.unique()):
            if idx == 0:
                dx = deformation_field_t[idx][:, :, 0] - coords[:, :, 0]  # shape [batch_size, flattenedpatch, ndims]
                dy = deformation_field_t[idx][:, :, 1] - coords[:, :, 1]
                dz = deformation_field_t[idx][:, :, 2] - coords[:, :, 2]
                similarity = (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3
            else:
                ncc = self.ncc_loss(warped_t[idx - 1], fixed)
                similarity += ncc

            if self.gradient_type == "analytic_gradient":
                spatial_smoothness_t, jac_det[idx] = self.smoothness.spatial(deformation_field_t[idx], coords)
            else:
                spatial_smoothness_t = self.smoothness.spatial(deformation_field_t[idx], coords)
                jac_det = None

            spatial_smoothness_t = spatial_smoothness_t * self.spatial_reg_weight
            spatial_smoothness += spatial_smoothness_t
            total_loss += (similarity + spatial_smoothness_t)

        temporal_smoothness = 0
        # temporal_smoothness = self.smoothness.temporal(self.batch_size, self.patch_size, deformation_field_t,
        #                                                self.time) * self.temporal_reg_weight
        total_loss += temporal_smoothness
        if jac_det != None:
            mono_loss = self.monotonic_constraint.forward(jac_det) * self.monotonicity_reg_weight
            total_loss += mono_loss

        print(
            "NCC: {}, Spatial smoothness: {}, Total loss: {}, Mono loss: {}, Temporal smoothness: {}".format(similarity, spatial_smoothness, total_loss,
                                                                            mono_loss, temporal_smoothness))
        return similarity, spatial_smoothness, total_loss, mono_loss, temporal_smoothness

    def mse_loss(self, predicted, target):
        return torch.mean((target - predicted) ** 2)

    def ncc_loss(self, warped_pixels, fixed_pixels):
        """
        This should be patchwise
        """
        nominator = ((fixed_pixels - fixed_pixels.mean()) *
                     (warped_pixels - warped_pixels.mean())).mean()
        denominator = fixed_pixels.std() * warped_pixels.std()

        cc = (nominator + 1e-6) / (denominator + 1e-6)
        return -torch.mean(cc)

    def time_mapping(self, out_features):
        hidden_features = 10
        mapping = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, out_features),
        )
        return mapping
