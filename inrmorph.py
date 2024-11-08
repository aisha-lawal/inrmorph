from settings import *
from utils import PositionalEncoding, SpatialTransform, SmoothDeformationField


##############################MAIN REGISTRATION CLASS#######################
class InrMorph(pl.LightningModule):
    def __init__(self, *args: Any):
        super().__init__()
        (self.I0, self.It, self.patch_size, self.spatial_reg_weight, self.temporal_reg_weight,
         self.batch_size, self.network_type, self.similarity_metric, self.gradient_type, self.loss_type, self.time) = args
        self.ndims = len(self.patch_size)
        self.nsamples = len(self.It)
        self.time_features = 64
        self.in_features = self.ndims
        self.out_features = self.ndims
        self.hidden_features = 256
        self.hidden_layers = 5
        self.omega = 30
        self.first_omega = 30
        self.hidden_omega = 30
        self.init_method = "sine"
        self.init_gain = 1
        self.fbs = 10 # k for bias initialization according to paper optimal

        assert self.loss_type in ["L1", "L2"], "Invalid loss type"
        assert self.gradient_type in ["finite_difference", "analytic_gradient"], "Invalid computation type"
        assert self.network_type in ["siren", "relu", "finer"], "Invalid network type"
        self.transform = SpatialTransform()
        self.t_mapping = self.time_mapping(self.time_features)
        self.smoothness = SmoothDeformationField(self.loss_type, self.gradient_type, self.patch_size,
                                                 self.batch_size)


        if self.network_type == "finer":
            
            self.mapping = Finer(in_features=self.in_features, out_features=self.out_features, hidden_layers=self.hidden_layers, hidden_features=self.hidden_features,
                      first_omega=self.first_omega, hidden_omega=self.hidden_omega,
                      init_method=self.init_method, init_gain=self.init_gain, fbs=self.fbs)

        elif self.network_type == "siren":
            
            self.mapping = Siren(layers=[self.in_features, *[self.hidden_features + (self.time_features * i) for i in
                                                             range(self.hidden_layers)], self.out_features],
                                 omega=self.omega, time_features=self.time_features)

        else:
            self.mapping = ReLU(layers=[self.in_features, *[self.hidden_features + i * (self.time_features * 2) for
                                                            i in range(self.num_layers)], self.out_features],
                                time_features=self.time_features)
    def forward(self, coords):

        displacement_t = [] #len samples

        for time in self.time: 
            t_n = time.unsqueeze(0).unsqueeze(0)
            
            t_n = t_n.expand(self.batch_size, coords.shape[1], 1)

            t_n = self.t_mapping(t_n) #concat this with every layer.
            phi_dims = self.mapping(coords, t_n) #single layer

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
        # if self.current_epoch % 50 == 0:

        #     self.visualize_mid_train()

        return val_loss

    def train_val_pipeline(self, coords: torch.Tensor, process: str):
        # flatten to shape [batch_size, flattened_patch_size, ndims]
        coords = coords.view(self.batch_size, np.prod(self.patch_size), self.ndims)
        coords = coords.clone().detach().requires_grad_(True)
        displacement_t = self.forward(coords)
        warped_t, fixed_t, deformation_field_t  = self.compute_transform(coords, displacement_t)
       
        ncc, smoothness, loss = self.compute_loss(warped_t, fixed_t, deformation_field_t, coords)

        self.log(f"{process}_ncc", ncc, on_epoch=True, sync_dist=True, on_step=False)
        self.log(f"{process}_spatial_smoothness", smoothness, on_epoch=True, sync_dist=True, on_step=False)
        self.log(f"{process}_loss", loss, on_epoch=True, sync_dist=True, on_step=False)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
   



    def compute_transform(self, coords, displacement):
        """
        Args:
        Here the baseline cooridnate (coords) serve as a reference coordinate which we align to. 
            - displacement (list of torch.tensor (*patch_size, ndims)): predicted displacement for each t in It, how  each point
            in the baseline should move to match the morphology of the followup at each point t
            - coords (torch.tensor): original coordinate used for warped
            - we warp the follow up image to the baseline coordinate for each t, to have a uniform coordinate system (baseline)
            warp function at time = 0 should be the identity transform

        """
        deformation_field_t = []
        warped_t = []
        fixed_t = []
        for idx, t in enumerate(self.time):
            deformation_field_t.append(torch.add(displacement[idx], coords)) #apply the displacement relative to the baseline coordinate (coords)
            warped_t.append(self.transform.trilinear_interpolation(coords=deformation_field_t[idx], img=self.I0).view(self.batch_size, *self.patch_size))
            fixed_t.append(self.transform.trilinear_interpolation(coords=coords, img=self.It[idx]).view(self.batch_size, *self.patch_size)) #resampling to the baseline coordinate

        return warped_t, fixed_t, deformation_field_t

    def compute_loss(self, warped_t, fixed_t, deformation_field_t, coords):
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
        total_loss = 0
        ncc = 0
        spatial_smoothness = 0
        for idx in range(self.nsamples):
            ncc_t = self.ncc_loss(warped_t[idx], fixed_t[idx])
            ncc += ncc_t
            spatial_smoothness_t = self.smoothness.spatial(deformation_field_t[idx], coords) * self.spatial_reg_weight
            spatial_smoothness += spatial_smoothness_t
            total_loss += (ncc_t + spatial_smoothness_t)
            # print("index and all loss", idx, ncc_t, spatial_smoothness_t, total_loss)
        #turn off spatial smoothness at epoch 300
        # if self.current_epoch >= 250:
        #     spatial_smoothness = 1e-6

       

        print("NCC: {}, Smoothness: {}, Total loss {}".format(ncc, spatial_smoothness, total_loss))
        return ncc, spatial_smoothness, total_loss

    def mse_loss(self, warped_pixels, fixed_pixels):
        return torch.mean((fixed_pixels - warped_pixels) ** 2)

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

    def visualize_mid_train(self):
        batch_size = 13520
        image_vector = CoordsImageTest(self.I0.shape, scale_factor = 1)
        test_generator = DataLoader(dataset = image_vector, batch_size=batch_size, shuffle = False)
        with torch.no_grad():

            for k, coords in enumerate(test_generator):
                coords = coords.squeeze().to(device, dtype=torch.float32)
                displacement_vector = self.test_step(coords)
                deformation_field = torch.add(displacement_vector, coords)
                deformation_field = deformation_field.cpu().detach()

                if k==0:

                    total_deformation_field = deformation_field
                else:
                    total_deformation_field = torch.cat((total_deformation_field, deformation_field), 0)

            temp_moved = self.transform.trilinear_interpolation(total_deformation_field, self.I0).view(self.I0.shape) 
            temp_moved = temp_moved.numpy().squeeze()

            self.wandb_log_images(self.It, self.I0, temp_moved, total_deformation_field, slice)
            del temp_moved


    def wandb_log_images(self, It, I0, moved, deformation_field_t, slice):
        images = {
                    "It": wandb.Image(It[0][:, :, slice].cpu().detach().numpy()),
                    "I0": wandb.Image(I0[0][:, :, slice].cpu().detach().numpy()),
                    "warped": wandb.Image(moved[0][:, :, slice].cpu().detach().numpy()),
                    "deformation_field": wandb.Image(deformation_field_t.view(self.batch_size, *self.patch_size, self.ndims)[0][:, :, slice, -1].cpu().detach().numpy())
                }
                
        wandb.log({"images": images})

# FINER

############### WEIGHTS INITIALIZATION ############################
# weights are initilized same as SIREN
def init_weights_cond(init_method, linear, omega=1, c=1, is_first=False):
    init_method = init_method.lower()
    if init_method == 'sine':
        init_weights(linear, omega, 6, is_first)  

def init_weights(m, omega=1, c=1, is_first=False): # Default: Pytorch initialization
    if hasattr(m, 'weight'):
        fan_in = m.weight.size(-1)
        if is_first: # 1/infeatures for first layer
            bound = 1 / fan_in 
        else:
            bound = np.sqrt(c / fan_in) / omega
        init.uniform_(m.weight, -bound, bound)

############### BIAS INITIALIZATION ############################
#bias are initialized as a uniform distribution between -k and k
def init_bias(m, k):
    if hasattr(m, 'bias'):
        init.uniform_(m.bias, -k, k)

def init_bias_cond(linear, fbs=None, is_first=True):
    if is_first and fbs != None:
        init_bias(linear, fbs)

############### FINER ACTIVATION FUNCTION ############################
# according to the paper, the activation function is sin(omega * alpha(x) * x), where alpha(x) = |x| + 1
def generate_alpha(x):
    with torch.no_grad():
        return torch.abs(x) + 1
    
def finer_activation(x, omega=1):
    return torch.sin(omega * generate_alpha(x) * x)


class FinerLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, omega=30, 
                 is_first=False, is_last=False, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None):
        super().__init__()
        self.omega = omega
        self.is_last = is_last ## no activation
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        # init weights
        init_weights_cond(init_method, self.linear, omega, init_gain, is_first)
        # init bias
        init_bias_cond(self.linear, fbs, is_first)
    
    def forward(self, input):
        wx_b = self.linear(input) 
        if not self.is_last:
            return finer_activation(wx_b, self.omega) #no activation for last layer
        return wx_b # is_last==True

      
class Finer(nn.Module):
    def __init__(self, in_features=3, out_features=3, hidden_layers=3, hidden_features=256, 
                 first_omega=30, hidden_omega=30, 
                 init_method='sine', init_gain=1, fbs=None, hbs=None):
        super().__init__()
        self.net = []
        self.net.append(FinerLayer(in_features, hidden_features, is_first=True, 
                                   omega=first_omega, 
                                   init_method=init_method, init_gain=init_gain, fbs=fbs))

        for i in range(hidden_layers):
            self.net.append(FinerLayer(hidden_features, hidden_features, 
                                       omega=hidden_omega, 
                                       init_method=init_method, init_gain=init_gain, hbs=hbs))

        self.net.append(FinerLayer(hidden_features, out_features, is_last=True, 
                                   omega=hidden_omega, 
                                   init_method=init_method, init_gain=init_gain, hbs=hbs)) # omega: For weight init
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)


##############SIREN####################

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
        coords = torch.sin(self.omega * self.layers[0](coords))
        
        for layer in self.layers[1:-1]:
            coords = torch.cat([coords, time], dim=-1)
            coords = torch.sin(self.omega * layer(coords))
        coords = torch.cat([coords, time], dim=-1)
        return self.layers[-1](coords)

################ReLU####################

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

class SegmentationLoss():
    pass