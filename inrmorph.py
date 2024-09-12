from settings import *
from utils import PositionalEncoding, SpatialTransform, SmoothDeformationField


##############################MAIN REGISTRATION CLASS#######################
class INRMorph(pl.LightningModule):
    def __init__(self, *args: Any):
        super().__init__()
        (self.I_0, self.I_t, self.patch_size, self.spatial_reg_weight, self.temporal_reg_weight,
         self.batch_size, self.network_type, self.similarity_metric, self.gradient_type, self.loss_type, self.time,
         self.observed_idx) = args
        self.len_time_seq = len(self.time)
        self.ndims = len(self.patch_size)

        self.time_features = 64
        self.in_features = self.ndims
        self.out_features = self.ndims

        self.hidden_features = 256

        self.num_layers = 5

        self.embedding_const = 10000

        self.positional_encoding = PositionalEncoding(self.len_time_seq, self.time_features,
                                                      self.embedding_const).encode()

        self.omega = 30
        assert self.loss_type in ["L1", "L2"], "Invalid loss type"
        assert self.gradient_type in ["finite_difference", "direct_gradient"], "Invalid computation type"
        assert self.network_type in ["siren", "relu"], "Invalid network type"
        self.transform = SpatialTransform()
        self.smoothness = SmoothDeformationField(self.loss_type, self.gradient_type, self.patch_size,
                                                 self.batch_size)
        self.samples = len(self.I_t)


        if self.network_type == "siren":

            self.mapping = Siren(layers=[self.in_features, *[self.hidden_features + (self.time_features * i) for i in
                                                             range(self.num_layers)], self.out_features],
                                 omega=self.omega, time_features=self.time_features)

            # self.mapping = Siren(layers=[self.in_features, *[self.hidden_features + i * (self.time_features * 2) for
            #                                                  i in range(self.num_layers)], self.out_features],
            #                      omega=self.omega, time_features=self.time_features)

        else:
            self.mapping = ReLU(layers=[self.in_features, *[self.hidden_features + i * (self.time_features * 2) for
                                                            i in range(self.num_layers)], self.out_features],
                                time_features=self.time_features)

        self.t_embedding = self.time_embedding(self.time_features)


    def forward(self, coords):

        displacement_t = torch.zeros(self.len_time_seq, self.batch_size, np.prod(self.patch_size), self.ndims).to(
            self.device)
        # pass single time for each point over the entire batch
        pos_enc = self.positional_encoding.unsqueeze(0).unsqueeze(0).permute(2, 0, 1,
                                                                             3)  # shape of [100, 1, 1, tim_features]
        time = self.time.expand(1, self.batch_size, coords.shape[1], self.time.shape[-1]).permute(3, 1, 2,
                                                                                                  0)  # shape of  [100, batch_size, patch_size, 1]
        time_embedding = self.t_embedding(time)  # shape of [100, batch_size, patch_size, time_features]
        time_embedding = time_embedding + pos_enc
        for t in range(self.len_time_seq):
            displacement_t[t] = self.mapping(coords, time_embedding[t])
        return displacement_t

    def training_step(self, batch, batch_idx):
        coords = batch
        train_loss = self.train_val_pipeline(coords, "train")

        return train_loss

    def validation_step(self, batch, batch_idx):
        # torch.set_grad_enabled(True)
        coords = batch
        val_loss = self.train_val_pipeline(coords, "val")

        return val_loss

    def train_val_pipeline(self, coords: torch.Tensor, process: str):
        # flatten to shape [batch_size, flattened_patch_size, ndims]
        coords = coords.view(self.batch_size, np.prod(self.patch_size), self.ndims)
        coords = coords.clone().detach().requires_grad_(True)
        displacement = self.forward(coords)
        warped_t, fixed_t, deformation_field_t = self.compute_transform(coords, displacement)
        loss_at_0, total_ncc, ncc_with_spatial_smoothness, spatial_smoothness, temporal_smoothness, loss = self.compute_loss(
            warped_t,
            fixed_t,
            deformation_field_t,
            coords)

        self.log(f"{process}_loss", loss, on_epoch=True, sync_dist=True, on_step=False)
        self.log(f"{process}_ncc_spatial_smoothness", ncc_with_spatial_smoothness, on_epoch=True, sync_dist=True,
                 on_step=False)
        self.log(f"{process}_spatial_smoothness", spatial_smoothness, on_epoch=True, sync_dist=True, on_step=False)
        self.log(f"{process}_temporal_smoothness", temporal_smoothness, on_epoch=True, sync_dist=True, on_step=False)
        self.log(f"{process}_ncc", total_ncc, on_epoch=True, sync_dist=True, on_step=False)
        self.log(f"{process}_loss_at_0", loss_at_0, on_epoch=True, sync_dist=True, on_step=False)

        return loss

    def test_step(self, batch, time):
        coords = batch
        batch_size = 1
        coords = coords.unsqueeze(0)
        tm = time.unsqueeze(0).unsqueeze(0)

        tm = tm.expand(batch_size, coords.shape[1], 1)

        tm = self.t_embedding(tm)
        displacement = self.mapping(coords, tm)

        return displacement


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        # add scheduler
        return optimizer

    def time_embedding(self, out_features):
        hidden_features = 10
        mapping = nn.Sequential(
            nn.Linear(1, hidden_features),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_features, out_features),
        )
        for module in [mapping]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    # init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu', a=0.2)
                    init.xavier_uniform_(layer.weight)
                    init.zeros_(layer.bias)
        return mapping

    def compute_transform(self, coords, displacement):
        """
        Args:
            displacement (list of torch.tensor (*patch_size, ndims)): predicted displacement for each t in I_t
            coords (torch.tensor): original coordinate used for warped
            warp function at time = 0 should be the identity transform
        """
        deformation_field_t = []  # len of displacement
        warped_t = []  # len observed samples - 1
        fixed_t = []  # len observed samples - 1
        for idx, time in enumerate(self.time):
            deformation_field_t.append(
                torch.add(displacement[idx], coords))  # shift coordinates [batch_size, flattened_patch_size, ndims]
            if idx != 0 and idx in self.observed_idx:
                warped_t.append(
                    self.transform.trilinear_interpolation(coords=deformation_field_t[self.observed_idx.index(idx)],
                                                           img=self.I_0).view(self.batch_size,
                                                                              *self.patch_size))  # [batch_size, *patch_size]

                fixed_t.append(self.transform.trilinear_interpolation(coords=coords,
                                                                         img=self.I_t[
                                                                             self.observed_idx.index(idx)]).view(
                    self.batch_size,
                    *self.patch_size))
        return warped_t, fixed_t, deformation_field_t  # 3, 3, 100

    def compute_loss(self, warped_t, fixed_t, deformation_field_t, coords):
        """
        Args: 
            warped_t (list of torch.Tensor): contains warped images at time t of shape
            [batch_size, *patch_size] i.e [batch_size, 32, 32, 32]

            fixed_t (list of torch.Tensor): list contains I_t at time t warped with original coords of shape 
            [batch_size, *patch_size] i.e [batch_size, 32, 32, 32]

            deformation_field_t (list of torch.Tensor): contains deformation field at time t of shape [batch_size, flattened_patch, ndims]

            coords: coordinats of shape [batch_size, flattened_patch, ndims]

        Returns: 

        """
        loss_at_t = 0
        total_spatial_smoothness = 0
        total_loss_with_spatial_smoothness_at_t = 0
        total_temporal_smoothness = 0
        count_idx = 0

        for idx, _ in enumerate(self.time):
            # similarity computation

            if idx == 0:
                dx = deformation_field_t[idx][:, :, 0] - coords[:, :, 0]  # shape [batch_size, flattenedpatch, ndims]
                dy = deformation_field_t[idx][:, :, 1] - coords[:, :, 1]
                dz = deformation_field_t[idx][:, :, 2] - coords[:, :, 2]

                loss_at_0 = (torch.mean(dx * dx) + torch.mean(dy * dy) + torch.mean(dz * dz)) / 3

            elif idx != 0 and idx in self.observed_idx:  # skip observed indices

                ncc_loss = self.ncc_loss(warped_t[count_idx], fixed_t[count_idx])
                loss_at_t += ncc_loss

                # spatial smoothness

                spatial_smoothness_at_t = self.smoothness.spatial(deformation_field_t[idx], coords)
                total_spatial_smoothness += spatial_smoothness_at_t * self.spatial_reg_weight

                # ncc with spatial smoothness
                total_loss_with_spatial_smoothness_at_t += (ncc_loss + (self.spatial_reg_weight *
                                                                        spatial_smoothness_at_t))
                count_idx += 1

            # else: #for unobserved samples
            #     spatial_smoothness_at_t = self.smoothness.spatial(deformation_field_t[idx], coords)
            #     total_spatial_smoothness += spatial_smoothness_at_t * self.spatial_reg_weight


        total_temporal_smoothness = (self.smoothness.temporal(deformation_field_t,
                                                              coords) / self.len_time_seq) * self.temporal_reg_weight
        total_loss = (loss_at_0 + (total_loss_with_spatial_smoothness_at_t /
                                   (self.samples - 1))) + total_temporal_smoothness

        print("Loss at 0: {}, Loss at t: {},  loss at t with Spatial Smoothness: {}, Spatial Smoothness: {}, TempralReg: {} -> Total loss: {} ".format(loss_at_0, loss_at_t/(self.samples -1),
                                                                                                                                                       total_loss_with_spatial_smoothness_at_t/(self.samples-1), total_spatial_smoothness/(self.samples-1), total_temporal_smoothness, total_loss))
        # wandb log image after 20 epochs
        return loss_at_0, loss_at_t / (self.samples - 1), total_loss_with_spatial_smoothness_at_t / (
                self.samples - 1), total_spatial_smoothness / (
                                  self.samples - 1), total_temporal_smoothness, total_loss

    def mse_loss(self, warped_pixels, fixed_pixels):
        return torch.mean((fixed_pixels - warped_pixels) ** 2)

    def ncc_loss(self, warped_pixels, fixed_pixels):

        nominator = ((fixed_pixels - fixed_pixels.mean()) *
                     (warped_pixels - warped_pixels.mean())).mean()

        denominator = fixed_pixels.std() * warped_pixels.std()

        cc = (nominator + 1e-6) / (denominator + 1e-6)

        return -torch.mean(cc)


###################SIREN ###############

# From INR IR paper
class Siren(nn.Module):

    def __init__(self, layers, omega, time_features):

        super(Siren, self).__init__()
        self.n_layers = len(layers) - 1
        self.omega = omega
        self.time_features = time_features

        # Make the layers
        self.layers = []
        for i in range(self.n_layers):
            with torch.no_grad():
                if i == 0:

                    self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                    self.layers[-1].weight.uniform_(-1 / layers[i], 1 / layers[i])
                else:
                    self.layers.append(nn.Linear(layers[i] + self.time_features, layers[i + 1]))
                    self.layers[-1].weight.uniform_(
                        -np.sqrt(6 / layers[i]) / self.omega,
                        np.sqrt(6 / layers[i]) / self.omega,
                    )

        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords, time):

        for layer in self.layers[:-1]:
            coords = torch.sin(self.omega * layer(coords))
            coords = torch.cat([coords, time], dim=-1)

        return self.layers[-1](coords)


class ReLU(nn.Module):
    def __init__(self, layers, time_features):
        super(ReLU, self).__init__()
        self.time_features = time_features
        self.layers = []

        for i in range(len(layers) - 1):
            if i == 0:
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            else:
                self.layers.append(nn.Linear(layers[i] + self.time_features, layers[i + 1]))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, coords, time):

        for layer in self.layers[:-1]:
            coords = torch.nn.functional.relu(layer(coords))
            coords = torch.cat([coords, time], dim=-1)

        return self.layers[-1](coords)
