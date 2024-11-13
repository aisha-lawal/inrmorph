import random
from settings import *


###################evaluation matrics/regularisation##################################

def spatial_reg(): #using diffusion reg
    pass

def temporal_reg(): #sobolev(try first order deriv, then second order for stricter penalty ), L1, L2, 
    pass


def mse_loss(*args: Any) -> int: #args = tuple of moving and fixed image    
    # if len(args) != 2:
    #     raise ValueError("Expecting two arguments:moving and fixed images")
    t0, tn, phi_dims, coords = args
    pass
    



################DATA LOADING AND PATCHING STUFF###########
    
def load_data(path: str, image) -> torch.tensor: #260, 260, 200
    data = np.array(nib.load(path).get_fdata())
    data = torch.tensor(data, device=device, dtype=torch.float32)
    if image: 
        # normalize for only images, not labels/masks
        return normalise(data)
    return data
    
def normalise(img: torch.Tensor) -> torch.Tensor:
    img = (img - img.min())/(img.max() - img.min())
    return (2 * img) - 1


def get_time_points(data):

    """
    Gets data path and returns time points between each image in months
    """
    data = sorted(glob.glob(data))
    splitpath = [data[i].split("/")[-1] for i in range(len(data))]
    dates = [datetime.strptime(img.split('_')[1] + '_' + img.split('_')[2], "%Y_%m") for img in splitpath]
    time_points = [0]+ [(dates[i+1].year - dates[i].year) * 12 + (dates[i+1].month - dates[i].month) for i in range(len(dates)-1)]

    return time_points

def define_coords(imgshape) -> torch.Tensor: 
    """
    defines coordinate between -1 to 1 of shape ndims
    returns tensor of shape (*imgshape, ndims)
    """
    ndims = len(imgshape)
    coords = [torch.linspace(-1, 1, imgshape[i])
               for i in range(ndims)]
    
    coords = torch.meshgrid(*coords, indexing=None)
    coords = torch.stack(coords, dim=ndims)

    return coords


# training is patchwise, val/test is full image!
class CoordsPatchesTrain(Dataset):
    def __init__(self, patch_size, npatches, imgshape):
        """doesnt account for non factors, look into later"""
    
        self.patch_size = patch_size
        assert len(self.patch_size) == 3 ,"incorrect patchsize, working with 3D data"

        self.imgshape = imgshape
        self.coords = define_coords(self.imgshape)
        self.npaches = npatches

        
        #adjust for ndims
        height, width, depth, dims = self.coords.size()
        patch_height, patch_width, patch_depth = patch_size
        num_patches_h = height // patch_height
        num_patches_w = width // patch_width
        num_patches_d = depth // patch_depth

        self.coords = self.coords.view(num_patches_h, patch_height, num_patches_w, 
                                       patch_width, num_patches_d, patch_depth, dims)
        # transpose and reshape the tensor to get patches as separate dimensions

        # self.coords = self.coords.permute(0, 2, 1, 3, 4).contiguous()
        self.coords = self.coords.permute(0, 2, 4, 1, 3, 5, 6).contiguous()

        self.coords = self.coords.view(num_patches_h * num_patches_w * num_patches_d, 
                                       patch_height, patch_width, patch_depth, dims)
    


    def __len__(self):
        return self.npaches

    def __getitem__(self, idx):
        # randomly sample npactches(with replacement)
        inds = torch.randint(0, len(self.coords), (self.npaches,))
        self.coords = self.coords[inds] #npatches, *patch_hwd, dims
        return self.coords[idx]



class CoordsPatch(Dataset):
    def __init__(self, patch_size, npatches, imgshape):
        # super(self, CoordsPatch).__init__()
        self.patch_size = np.ceil(np.array(patch_size)/2).astype(np.int16)    
        self.ndims = len(self.patch_size)
        self.imgshape = imgshape
        self.coords = define_coords(self.imgshape)
        self.dx = torch.div(2, torch.tensor(self.coords.shape[:-1]) )
        self.npatches = npatches #how many random patches to sample

        self.patch_size = np.ceil(np.array(patch_size)/2).astype(np.int16) 
        patch_dx_dims = torch.tensor(self.patch_size) * self.dx 
        
        patch_coords = [torch.linspace(-patch_dx_dims[i], patch_dx_dims[i], 2*self.patch_size[i]) for i in range(self.ndims)]
        patch_coords = torch.meshgrid(*patch_coords, indexing=None)
        self.patch_coords = torch.stack(patch_coords, dim=self.ndims)  

        coords = self.coords[self.patch_size[0]:-self.patch_size[0], 
                        self.patch_size[1]:-self.patch_size[1], self.patch_size[2]:-self.patch_size[2], ...]


        self.spatial_size = coords.shape[:-1]                                            
        self.coords = coords

    def __len__(self):
        return self.npatches
    
    def __getitem__(self, idx):
        indx = np.random.randint(0, np.prod(self.spatial_size))
        
        inds = np.unravel_index(indx, self.spatial_size)
        
        center = self.coords[inds[0], inds[1], inds[2],  :]
        coords = torch.clone(self.patch_coords)
        
        coords[..., 0] = coords[..., 0] + center[0] 
        coords[..., 1] = coords[..., 1] + center[1]
        coords[..., 2] = coords[..., 2] + center[2]
        return coords

class CoordsImageTest(Dataset):
    def __init__(self, imgshape, scale_factor = 1):
        """
        input: coords with shape (img_height, img_width, ndims)
        Returns: flatten the image where each point has ndims, returns shape (imgshape, ndims)
        """
        self.imgshape = imgshape
        self.coords = define_coords(self.imgshape)
        self.scale_factor = scale_factor
        
        if self.scale_factor!=1: #to test with larger resolution
            self.coords = F.interpolate(self.coords.permute(3, 2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
            self.coords = self.coords.squeeze().permute(2, 3, 1, 0)

            # coords = F.interpolate(coords.permute(2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            # coords = coords.squeeze().permute(1, 2, 0)

        self.ndims = self.coords.shape[-1]
        self.coords = self.coords.view([np.prod(self.coords.shape[:-1]), self.ndims])

    def __len__(self):
        return self.coords.shape[0] #flattened imgshape
    
    def __getitem__(self, idx):
        return self.coords[idx, :]


##########SPATIAL TRANSFORM#############

class SpatialTransform():
    def __init__(self) -> torch.Tensor:
        """
        Input: takes in predicted deformation at point x,y,z and input coordinates x,y,z,
            transforms predicted deformation to same coordinate space as coord(input coord). We
            derive the "shifted_coords" from this.
            This means the displacement field that moves image/point A to B will be "in the space of B"!!
        
        Interpolation: For each coordinate point in "shifted_coords", we need to derive the pixel 
            intensity values for each coordinate. We derive these pixel values from the moving image. So 
            basically we transform the coordinates of the pixels in the moving image to that of the 
            "shifted_coords". To derive these pixel values, we do interpolation for each point on the 

        """
        

    def bilinear_interpolation(self,coords: torch.Tensor, img: torch.Tensor):
        

        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]

        #rescale coords from to range between 0 and imgshape -1, useful since x_coords and y_coords
        # were initially between -1 and 1 
        x_coords = (x_coords + 1) * (img.shape[0] - 1) * 0.5
        y_coords = (y_coords + 1) * (img.shape[1] - 1) * 0.5

        x0 = torch.floor(x_coords).to(torch.long)
        y0 = torch.floor(y_coords).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1

        # we have to clamp to ensure the nearest neighbours are within the boundaries of the image, 
        #if not we may not find that particular point
        x0 = torch.clamp(x0, 0, img.shape[0] - 1)
        x1 = torch.clamp(x1, 0, img.shape[0] - 1)
        y0 = torch.clamp(y0, 0, img.shape[1] - 1)
        y1 = torch.clamp(y1, 0, img.shape[1] - 1)

        # r1 = img[x0, y0] * ((x1 - x_coords)/(x1 - x0)) + img[x1, y0] * ((x_coords - x0)/(x1 - x0)) #same as representing the first weight as (1 - second weight)
        # r2 = img[x0, y1] * ((x1 - x_coords)/(x1 - x0)) + img[x1, y1] * ((x_coords - x0)/(x1 - x0))
        # pixel_values = r1 * ((y1 - y_coords)/(y1 - y0)) + r2 * ((y_coords - y0)/(y1 - y0))

      
        x_coords = x_coords - x0
        y_coords = y_coords - y0

        pixel_values = (img[x0, y0] * (1 - x_coords) * (1 - y_coords) 
                + img[x1, y0] * x_coords * (1 - y_coords) 
                + img[x0, y1] * (1 - x_coords) * y_coords
                + img[x1, y1] * x_coords * y_coords) 
        
        # print("in interpolation", pixel_values.shape) #[batch_size, flattened_patchsize]
      
        return pixel_values 
       
    #from INR paper
    def trilinear_interpolation(self,coords, img): 
        """
        Args: coords of shape [batchsize, flattened_patchsize, ndims]
            img of shape [h,w,d]
        
        
        """
        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]
        z_coords = coords[:, :, 2]
        
        x_coords = (x_coords + 1) * (img.shape[0] - 1) * 0.5 
        y_coords = (y_coords + 1) * (img.shape[1] - 1) * 0.5
        z_coords = (z_coords + 1) * (img.shape[2] - 1) * 0.5

        x0 = torch.floor(x_coords.detach()).to(torch.long)
        y0 = torch.floor(y_coords.detach()).to(torch.long) 
        z0 = torch.floor(z_coords.detach()).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1
        z1 = z0 + 1

        x0 = torch.clamp(x0, 0, img.shape[0] - 1)
        y0 = torch.clamp(y0, 0, img.shape[1] - 1)
        z0 = torch.clamp(z0, 0, img.shape[2] - 1)
        x1 = torch.clamp(x1, 0, img.shape[0] - 1)
        y1 = torch.clamp(y1, 0, img.shape[1] - 1)
        z1 = torch.clamp(z1, 0, img.shape[2] - 1)

        x = x_coords - x0
        y = y_coords - y0
        z = z_coords - z0

        pixel_values = (
            img[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
            + img[x1, y0, z0] * x * (1 - y) * (1 - z)
            + img[x0, y1, z0] * (1 - x) * y * (1 - z)
            + img[x0, y0, z1] * (1 - x) * (1 - y) * z
            + img[x1, y0, z1] * x * (1 - y) * z
            + img[x0, y1, z1] * (1 - x) * y * z
            + img[x1, y1, z0] * x * y * (1 - z)
            + img[x1, y1, z1] * x * y * z
        )
        
        return pixel_values

    def nearest_neighbor_interpolation(self, coords, img):
        """
        Args: coords of shape [batchsize, flattened_patchsize, ndims]
            img of shape [h,w,d]
        """
        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]
        z_coords = coords[:, :, 2]
        
        x_coords = (x_coords + 1) * (img.shape[0] - 1) * 0.5 
        y_coords = (y_coords + 1) * (img.shape[1] - 1) * 0.5
        z_coords = (z_coords + 1) * (img.shape[2] - 1) * 0.5

        x0 = torch.round(x_coords.detach()).to(torch.long)
        y0 = torch.round(y_coords.detach()).to(torch.long) 
        z0 = torch.round(z_coords.detach()).to(torch.long)

        x0 = torch.clamp(x0, 0, img.shape[0] - 1)
        y0 = torch.clamp(y0, 0, img.shape[1] - 1)
        z0 = torch.clamp(z0, 0, img.shape[2] - 1)

        pixel_values = img[x0, y0, z0]
       
        return pixel_values
    
    def spline(self,):
        pass


#####################SMOOTH FIELD###############
class SmoothDeformationField():
    """
    Encourages smoothness, we compute the graident using finite difference and take the L2/L1 , penalize more using L2
        -> by calculating absolute squared differences between neighbouring elements along dimensions and averaging, L1.

    Input: shape of [batch_size, flattened_patchsize, ndims]
    """
    def __init__(self, loss_type, gradient_type, patch_size, batch_size):

        self.loss_type = loss_type
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.gradient_type = gradient_type
        self.gradient_computation = GradientComputation()


    def spatial(self, field, coords):
        #using anaytic gradient, computing the derivates of field wrt coords
        if self.gradient_type == "analytic_gradient":
            # jacobian_matrix = self.gradient_computation.compute_matrix(coords, field)
            gradients = self.gradient_computation.gradient(coords, field)
            l2_norm = torch.norm(gradients, dim=-1, p=2).mean()
            return l2_norm
        

        #field is of shape [batch_size, *patch_size, ndims], use finite difference approximation
        else:
            field = field.view(self.batch_size, *self.patch_size, len(self.patch_size))
            spacing = 1 #spacing mm along x, y and z
            x = field[:, :, :, :, 0]
            y = field[:, :, :, :, 1]
            z = field[:, :, :, :, 2]

            # compute gradients along each axis (x, y, z) for each displacement component, retuns a tuple of gradients along each axis
            gradients_x = torch.gradient(x, dim=(1, 2, 3), spacing=1)
            gradients_y = torch.gradient(y, dim=(1, 2, 3), spacing=1)
            gradients_z = torch.gradient(z, dim=(1, 2, 3), spacing=1)

            #sum of mean squared gradients
            smoothness = sum((grad**2).mean() for grad in gradients_x + gradients_y + gradients_z)

            return smoothness
           
            # du = torch.gradient(field, spacing=1, dim=(1, 2, 3))
            # dx = du[0]
            # dy = du[1]
            # dz = du[2]
            # return (dx*dx).mean() + (dy*dy).mean() + (dz*dz).mean()

    def temporal(self, field_t, time_points):
        """
        field_t: list of fields at different time points, each field is of shape [batch_size, *patch_size, ndims]
        time_points: list of time points, len(time_points) = len(field_t) + 1

        We compute:
            d_phi/dt = (phi_t+1 - phi_t)/delta_t
        """
    
        field_t = [field.view(self.batch_size, *self.patch_size, 
                            len(self.patch_size)) for field in field_t]        
        field_t = [field.unsqueeze(0) for field in field_t]
        field_t = torch.cat(field_t, dim=0) 

        delta_t = time_points[1:] - time_points[:-1] #compute_difference between consecutive time points
        # u_diff = field_t[1:] - field_t[:-1] 
        
        # delta_t = delta_t.view(-1, 1, 1, 1, 1) 
        
        # temp_grad = u_diff / delta_t 
        temp_grad = torch.gradient(field_t, delta_t=(time_points,), dim=0)  
        print(temp_grad.shape)
        tempreg = (temp_grad ** 2).mean() 
        return tempreg

            


    # def temporal(self, field_t, coords):
    #     if self.gradient_type == "analytic_gradient":
            
    #         coords = coords.unsqueeze(0)
    #         coords = [coords for _ in range(len(field_t))]
    #         coords = torch.cat(coords, dim=0)


    #         field_t = [field.unsqueeze(0) for field in field_t]
    #         field_t = torch.cat(field_t, dim=0) 

    #         gradients = self.gradient_computation.gradient(coords, field_t)
    #         l2_norm = torch.norm(gradients, dim=-1, p=2).mean()

    #         return l2_norm

    #     else:
    #         field_t = [field.view(self.batch_size, *self.patch_size, 
    #                             len(self.patch_size)) for field in field_t]        
    #         field_t = [field.unsqueeze(0) for field in field_t]
    #         field_t = torch.cat(field_t, dim=0) 
    #         if self.loss_type == "L1":
    #             dt = torch.abs(field_t[1:, :, :, :, :, :] - field_t[:-1, :, :, :, :, :])
    #             # dt = torch.diff(field_t, n=1, dim=0)
    #             return torch.mean(dt)
            
    #         elif self.loss_type == "L2":
    #             dt = field_t[1:, :, :, :, :, :] - field_t[:-1, :, :, :, :, :]

    #             # dt = torch.diff(field_t, n=2, dim=0)
    #             return torch.mean(dt * dt)
            

  

class GradientComputation():
    def __init__(self):
        pass

    def compute_matrix(self, coords, field):
        """
        compute partial derivatives for each dimension of the field wrt each dimension of the coords.
        useful for computing jacobian matrix
        
        """
        dim = coords.shape[-1]
        patch = coords.shape[1]
        batch_size = coords.shape[0]
        # print("in jac", coords.shape, field.shape, dim, coords.requires_grad, field.requires_grad)
        matrix = torch.zeros(batch_size, patch, dim, dim)

        for b in range(batch_size):
            # print("in jac checking batch", coords[b].shape, field[b].shape, dim, coords.requires_grad, field.requires_grad)
            for i in range(dim):
                
                matrix[b, :, i, :] = self.gradient(coords, field[b, :,  i], b) # for partial derivatives, wrt x,yz

        return matrix   
    
    def gradient(self, input_coords, output, b=None, grad_outputs=None):
        # print("in grad", input_coords.shape, output.shape, b, input_coords.requires_grad, output.requires_grad)

        grad_outputs = torch.ones_like(output)

        grad= torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        if b == None:
            return grad
        else:
            return grad[b]
    
class FiniteDifference():
    def __init__(self):
        pass

    def compute(self, field):
        """"
        Takes as input field of shape [batch_size, *patch_size, ndims]
        """
        field = field.view(self.batch_size, *self.patch_size, len(self.patch_size))
        spacing = 1 #spacing along x, y and z
        x = field[:, :, :, :, 0]
        y = field[:, :, :, :, 1]
        z = field[:, :, :, :, 2]

        # compute gradients along each axis (x, y, z) for each displacement component, retuns a tuple of gradients along each axis
        gradients_x = torch.gradient(x, dim=(1, 2, 3), spacing=spacing)
        gradients_y = torch.gradient(y, dim=(1, 2, 3), spacing=spacing)
        gradients_z = torch.gradient(z, dim=(1, 2, 3), spacing=spacing)

        #sum of mean squared gradients
        smoothness = sum((grad**2).mean() for grad in gradients_x + gradients_y + gradients_z)
        return smoothness

        # du = torch.gradient(field, spacing)
        # dx = du[0]
        # dy = du[1]
        # dz = du[2]
        # return (dx*dx).mean() + (dy*dy).mean() + (dz*dz).mean()

class PenalizeLocalVolumeChange():
    """
    penalizes local volume change
    """
    def __init__(self):
        pass
    def forward(self, coords, field):
        ndims = coords.shape[-1] 

        jacobian_matrix = torch.zeros(coords.shape[0], ndims, ndims)

        for i in range(ndims):

            jacobian_matrix[:, i, :] = self.gradient(coords, field[:, i]) #derivative of field wrt each dimension in coords
            jacobian_matrix[:, i, i] += torch.ones_like(jacobian_matrix[:, i, i]) #add ones on diagonal to preserve local volume across diagnoal elements

        loss = 1 - torch.det(jacobian_matrix)
        loss = torch.mean(torch.abs(loss)) #absolute deviation between 1(unity determinant- i.e no volume change) and |J|
        return loss
    
    def gradient(self, coords, field):
        gradient_outputs = torch.ones_like(field)
        gradient = torch.autograd.grad(field, [coords], grad_outputs=gradient_outputs, create_graph=True)[0]
        return gradient 


#########################TIME SEQUENCE GENERATION#########################
def generate_random_sum(len_time_seq, n):
    cut_points = sorted(random.sample(range(len_time_seq), n - 1))
    values = [cut_points[0]] + [cut_points[i] - cut_points[i - 1] 
                                for i in range(1, n - 1)] + [len_time_seq - cut_points[-1]]
    return values

def generate_time_sequence(time_points, random_indexes):
    time_seq = []
    for i in range(len(time_points)-1):
        random_values = np.random.uniform(time_points[i], time_points[i+1], random_indexes[i]-2)
        random_values = np.concatenate(([time_points[i]], np.sort(random_values), [time_points[i+1]]))
        time_seq = np.concatenate((time_seq, random_values))
    time_seq = np.unique(time_seq)  
    return time_seq
    

# positional encoding
class PositionalEncoding:
    """
    Each position should be the same size as the output of the time embedding since we are summing them up
    """
    def __init__(self, len_token_index=100, dim_out_embs=64, n=10000):
        self.dim_out_embs = dim_out_embs
        self.n = n
        self.len_token_index = len_token_index #len_token_index == len_time_object

    def encode(self): 
        pos_enc = torch.zeros(self.len_token_index, self.dim_out_embs).to(device)
        for token_index in range(self.len_token_index):
            for i in np.arange(int(self.dim_out_embs/2)):
                denominator = np.power(self.n, 2*i/self.dim_out_embs)
                pos_enc[token_index, 2*i] = np.sin(token_index/denominator)
                pos_enc[token_index, 2*i+1] = np.cos(token_index/denominator)
        return pos_enc