from typing import List
import torch.nn as nn
from torch import Tensor
from config import device, NetworkType, GradientType, SimilarityMetric, SpatialRegularizationType
import numpy as np
import torch
from models.finer import Finer
from models.relu import ReLU
from models.siren import Siren
from utils import SpatialTransform, SpatioTemporalRegularization, MonotonicConstraint
import lightning as pl
from torch.optim.lr_scheduler import StepLR, LambdaLR
from pytorch_lightning.callbacks import ModelCheckpoint



class InrMorph(pl.LightningModule):
    def __init__(self,
                 I0: Tensor,
                 It: List,
                 patch_size: List,
                 spatial_reg_weight: float,
                 temporal_reg_weight: float,
                 monotonicity_reg_weight: float,
                 spatial_reg_type: str,
                 batch_size: int,
                 network_type: str,
                 similarity_metric: str,
                 gradient_type: str,
                 time: Tensor,
                 observed_time_points: Tensor,
                 lr: float,
                 weight_decay: float,
                 omega_0: float,
                 hidden_layers: int,
                 time_features: int,
                 hidden_features: int,
                 num_epochs: int,
                 extrapolate: bool,
                 l2_weight: float,
                 fixed_time_embedding: bool
                 ) -> None:
        super().__init__()
        self.I0 = I0
        self.It = It
        self.patch_size = patch_size
        self.spatial_reg_weight = spatial_reg_weight
        self.temporal_reg_weight = temporal_reg_weight
        self.monotonicity_reg_weight = monotonicity_reg_weight
        self.spatial_reg_type = spatial_reg_type
        self.batch_size = batch_size
        self.network_type = network_type
        self.similarity_metric = similarity_metric
        self.gradient_type = gradient_type
        self.observed_time_points = observed_time_points
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.extrapolate = extrapolate
        self.l2_weight = l2_weight
        self.fixed_time_embedding = fixed_time_embedding

        self.ndims = len(self.patch_size)
        self.nsamples = len(self.It)
        self.time_features = time_features
        self.in_features = self.ndims
        self.out_features = self.ndims
        self.hidden_features = hidden_features
        self.hidden_layers = hidden_layers
        self.omega_0 = omega_0
        # self.seed = 42
        self.flattened_patch_size = np.prod(self.patch_size)

        #reshaping time, this is needed so we can compute voxelwise derivative for temporal and mono smoothness
        # self.time = self.time.clone().detach().requires_grad_(True)
        time = time.view(-1, 1, 1, 1, 1)
        time = time.expand(-1, self.batch_size, *self.patch_size)
        self.time = time.clone().detach().requires_grad_(True)
        self.first_omega = 30
        self.hidden_omega = 30
        self.init_method = "sine"
        self.init_gain = 1
        self.fbs = 5  # k for bias initialization according to paper optimal
        self.transform = SpatialTransform()


        
        self.smoothness = SpatioTemporalRegularization(self.gradient_type, self.patch_size,
                                                 self.batch_size, self.time, self.spatial_reg_type)

        self.monotonic_constraint = MonotonicConstraint(self.patch_size, self.batch_size, self.time, self.gradient_type)


        if self.fixed_time_embedding:
            pass
        else:
            self.t_mapping = self.time_mapping(self.time_features)
        if self.network_type == NetworkType.FINER:

            self.mapping = Finer(in_features=self.in_features, out_features=self.out_features,
                                 hidden_layers=self.hidden_layers, hidden_features=self.hidden_features,
                                 first_omega=self.first_omega, hidden_omega=self.hidden_omega,
                                 init_method=self.init_method, init_gain=self.init_gain, fbs=self.fbs)

        elif self.network_type == NetworkType.SIREN:
            

            self.mapping = Siren(layers=[self.in_features, *[self.hidden_features + (self.time_features * i) for i in
                                                             range(self.hidden_layers)], self.out_features],
                                 omega_0=self.omega_0, time_features=self.time_features)

        else:
            self.mapping = ReLU(layers=[self.in_features, *[self.hidden_features + i * (self.time_features * 2) for
                                                            i in range(self.hidden_layers)], self.out_features],
                                time_features=self.time_features)


    def forward(self, coords):
        displacement_t = []  # len samples
        for idx, time in enumerate(self.time):
            t_n = time.view(self.batch_size, self.flattened_patch_size, 1) 
            t_n = self.t_mapping(t_n)  # concat this with every hidden layer of the INR.
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
        coords = coords.view(self.batch_size, self.flattened_patch_size, self.ndims)
        coords = coords.clone().detach().requires_grad_(True)
        displacement_t = self.forward(coords)
        warped_t, fixed, deformation_field_t = self.compute_transform(coords, displacement_t)

        similarity_at_0, ncc, spatial_smoothness, loss, mono_loss, temporal_smoothness = self.compute_loss(warped_t,
                                                                                          fixed, deformation_field_t, coords)

        metrics = {
            f"{process}_similarity_at_0": similarity_at_0,
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

   

    def compute_transform(self, coords, displacement):
        """
        Args:
        Here the baseline cooridnate (coords) serve as a reference coordinate which we align to. 
            - displacement (list of torch.tensor (*patch_size, ndims)): predicted displacement for each t in It, how each location in coords
            It should move to align with I0 (baseline)
            - coords (torch.tensor): original coordinate used for warped
            - we warp the follow up image to the base line coordinate for each t, to have a uniform coordinate system (baseline)
            warp function at time = 0 should be the identity transform so we compute L2 norm of the displacement at time 0

        """
        deformation_field_t = []
        warped_t = []
        fixed = self.transform.trilinear_interpolation(coords=coords, img=self.I0).view(self.batch_size,
                                                                                        *self.patch_size)  # to reshape as a patch
        for idx, t in enumerate(self.time.unique()):
            deformation_field_t.append(torch.add(displacement[idx],
                                                 coords))  # apply the displacement relative to the baseline coordinate (coords)
            if idx != 0 and t in self.observed_time_points: #we dont warp extrapolated and interpolated points
                #get timepoint ids relative to the displacement field and observed ids
                # indexes arent same because we have to consider the timepoints we are observing 
                disp_id = self.time.unique().tolist().index(t)
                observed_id = self.observed_time_points.index(t)
                # print(f"id is {disp_id, observed_id}, tm is {t}")
                warped_t.append(
                    self.transform.trilinear_interpolation(coords=deformation_field_t[disp_id], img=self.It[observed_id]).view(
                        self.batch_size, *self.patch_size))
            # print(f"warped_t length is {len(warped_t), len(deformation_field_t)}")
        return warped_t, fixed, deformation_field_t

    def compute_loss(self, warped_t, fixed, deformation_field_t, coords):
        """
        Args: 
            warped_t (list of torch.Tensor):  warped images at time t of shape (except interpolated and interpolated points)
            [batch_size, *patch_size] i.e [batch_size, 32, 32, 32]

            fixed_t (list of torch.Tensor): baseline coords patchsized
            [batch_size, *patch_size] i.e [batch_size, 32, 32, 32]

            deformation_field_t (list of torch.Tensor):  deformation field at time t of shape [batch_size, flattened_patch, ndims]

            coords: coordinates of shape [len(time), batch_size, flattened_patch, ndims] 

        Returns: 

        """
        mono_loss = 0
        total_loss = 0
        similarity = 0
        temporal_smoothness = 0
        spatial_smoothness = 0
        jac_det = torch.zeros(len(self.time.unique()), self.batch_size, self.flattened_patch_size).to(device)
        for idx, tm in enumerate(self.time.unique()):
            if idx == 0:
                dx = deformation_field_t[idx][:, :, 0] - coords[:, :, 0]  # shape [batch_size, flattenedpatch, ndims]
                dy = deformation_field_t[idx][:, :, 1] - coords[:, :, 1]
                dz = deformation_field_t[idx][:, :, 2] - coords[:, :, 2]
                similarity_at_0 = self.l2_weight * torch.mean(torch.sqrt(dx**2 + dy**2 + dz**2)) #l2 norm

            #for extrapolated point and interpolated points compute the regularization alone, 
            #we dont observe data at this point but we can generate a deformation field
            elif tm not in self.observed_time_points:
                similarity_t = 0
            else:
                observed_id = self.observed_time_points.index(tm)
                ncc = self.ncc_loss(warped_t[observed_id - 1], fixed) #we start from 1 because we dont have warped_t at 0
                similarity_t = ncc

            similarity += similarity_t # for NCC plot
            
            if self.gradient_type == GradientType.ANALYTIC_GRADIENT:
                #spatial
                spatial_smoothness_t, jac_det[idx] = self.smoothness.spatial(deformation_field_t[idx], coords)
                # spatial_smoothness_t = self.smoothness.spatial(deformation_field_t[idx], coords) #no monoloss

            else:
                spatial_smoothness_t = self.smoothness.spatial(deformation_field_t[idx], coords)
                jac_det = None
            #condition for spatial reg type for each timepoint
            if self.spatial_reg_type == SpatialRegularizationType.SPATIAL_JACOBIAN_MATRIX_PENALTY:
                spatial_smoothness_t = spatial_smoothness_t * self.spatial_reg_weight
                spatial_smoothness += spatial_smoothness_t
                total_loss += (similarity_t + spatial_smoothness_t)
                # total_loss += (similarity_t) #no spatial smoothness
            else:
                #if spatial_reg_type is smoothness of rate of change, don't calculate independent smoothness
                total_loss += similarity_t
    
            # if self.optimizers().param_groups[0]['lr'] <= 1e-5:
            if self.current_epoch > 200: #start temporal and mono smoothness after 60 epochs
                #condition for spatial smoothness in temporal rate of change
                if self.spatial_reg_type == SpatialRegularizationType.SPATIAL_JACOBIAN_MATRIX_PENALTY:
                    temporal_smoothness = self.smoothness.temporal(deformation_field_t, coords) * self.temporal_reg_weight
                    total_loss += temporal_smoothness
                else:
                    #spatial_smoothness below overrides the defined one above
                    temporal_smoothness, spatial_smoothness = self.smoothness.temporal(deformation_field_t, coords)
                    temporal_smoothness = temporal_smoothness * self.temporal_reg_weight
                    spatial_smoothness = spatial_smoothness * self.spatial_reg_weight
                    # temporal_smoothness = 0 #no temporal
                    total_loss += temporal_smoothness + spatial_smoothness

                if jac_det != None: #will be None if gradient type is numerical approx
                    mono_loss = self.monotonic_constraint.forward(jac_det) * self.monotonicity_reg_weight
                    total_loss += mono_loss
        print(
            "Similarity at 0: {}, NCC: {}, Spatial smoothness: {}, Total loss: {}, Mono loss: {}, Temporal smoothness: {}".format(similarity_at_0, similarity, spatial_smoothness, total_loss,
                                                                            mono_loss, temporal_smoothness))
        return similarity_at_0, similarity, spatial_smoothness, total_loss, mono_loss, temporal_smoothness

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
