import numpy as np
import torch
from config import device


##########SPATIAL TRANSFORM#############

class SpatialTransform:
    def __init__(self) -> None:
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

    def bilinear_interpolation(self, coords: torch.Tensor, img: torch.Tensor):
        x_coords = coords[:, :, 0]
        y_coords = coords[:, :, 1]

        # rescale coords from to range between 0 and imgshape -1, useful since x_coords and y_coords
        # were initially between -1 and 1
        x_coords = (x_coords + 1) * (img.shape[0] - 1) * 0.5
        y_coords = (y_coords + 1) * (img.shape[1] - 1) * 0.5

        x0 = torch.floor(x_coords).to(torch.long)
        y0 = torch.floor(y_coords).to(torch.long)
        x1 = x0 + 1
        y1 = y0 + 1

        # we have to clamp to ensure the nearest neighbours are within the boundaries of the image,
        # if not we may not find that particular point
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

        # from INR paper

    def trilinear_interpolation(self, coords: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
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

    def nearest_neighbor_interpolation(self, coords: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
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

    def spline(self, ):
        pass


#####################Regularization###############
class SmoothDeformationField:
    """
    Input: shape of [batch_size, flattened_patchsize, ndims]
    """

    def __init__(self, gradient_type, patch_size, batch_size):

        self.patch_size = patch_size
        self.batch_size = batch_size
        self.gradient_type = gradient_type
        self.gradient_computation = GradientComputation()

    def spatial(self, field, coords):
        if self.gradient_type == "analytic_gradient":
            jacobian_matrix = self.gradient_computation.compute_matrix(coords, field)

            # can also compute frobenius norm of jacobian matrix i.e L2 norm in matrix form.
            l2 = torch.norm(jacobian_matrix, dim=(-2, -1), p=2).sum(dim=1).mean()
            # smoothness_loss = torch.sum(jacobian_matrix **2, dim=[1, 2, 3]).mean()
            #compute jacobian determinant component
            jacobian_determinants = self.compute_jacobian_determinant(jacobian_matrix)
            return l2, jacobian_determinants

        else:
            field = field.view(self.batch_size, *self.patch_size, len(self.patch_size))
            spacing = 1
            x = field[:, :, :, :, 0]
            y = field[:, :, :, :, 1]
            z = field[:, :, :, :, 2]

            gradients_x = torch.gradient(x, dim=(1, 2, 3), spacing=spacing)
            gradients_y = torch.gradient(y, dim=(1, 2, 3), spacing=spacing)
            gradients_z = torch.gradient(z, dim=(1, 2, 3), spacing=spacing)
            smoothness_loss = sum((grad ** 2).mean() for grad in gradients_x + gradients_y + gradients_z)
            # fieldx = torch.sum((field[:, 1:, :-1, :-1, :] - field[:, :-1, :-1, :-1, :]) ** 2, dim=(1, 2, 3, 4)).mean()
            # fieldy = torch.sum((field[:, :-1, 1:, :-1, :] - field[:, :-1, :-1, :-1, :]) ** 2, dim=(1, 2, 3, 4)).mean()
            # fieldz = torch.sum((field[:, :-1, :-1, 1:, :] - field[:, :-1, :-1, :-1, :]) ** 2, dim=(1, 2, 3, 4)).mean()
            # smoothness_loss= fieldx + fieldy + fieldz
            return smoothness_loss

    def temporal(self, batch_size, patch_size, field_t, time):
        """
        we do dfield/dt
        field is of shape [batch_size, flattened_patchsize, ndims] and len(time.unique())
        """
        dims = len(patch_size)
        field_t = [field.view(batch_size, *patch_size, dims) for field in field_t]
        field_t = torch.stack(field_t, dim=0)
        if self.gradient_type == "analytic_gradient":
            dfield_dt = torch.autograd.grad(
                outputs=field_t,
                inputs=time,
                grad_outputs=torch.ones_like(field_t),
                create_graph=True,
            )[0]
        else:
            #for numerical approximation
            dfield_dt = (field_t[1:] - field_t[:-1])/ (time[1:] - time[:-1])

        temporal_smoothness = torch.sum(dfield_dt**2, dim=[2, 3, 4, 5])
        temporal_smoothness = temporal_smoothness.mean(dim=1) #batch_size
        # return temporal_smoothness.mean() similarity not averaged
        return temporal_smoothness

    
    def compute_jacobian_determinant(self, jacobian_matrix):
        #shape of [batch_size, flattened_patch_size, ndims, ndims]
        # for 3x3 using det(A)=a(ei−fh)−b(di−fg)+c(dh−eg); (i.e compute 2D det and multiply with held out col/row)
        # a = jacobian_matrix[:, :, 0, 0]
        # b = jacobian_matrix[:, :, 0, 1]
        # c = jacobian_matrix[:, :, 0, 2]
        # d = jacobian_matrix[:, :, 1, 0]
        # e = jacobian_matrix[:, :, 1, 1]
        # f = jacobian_matrix[:, :, 1, 2]
        # g = jacobian_matrix[:, :, 2, 0]
        # h = jacobian_matrix[:, :, 2, 1]
        # i = jacobian_matrix[:, :, 2, 2]
        # determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

        #make above tidy
        #first second and third row i.e a,b,c above
        dx = jacobian_matrix[..., 0, :]
        dy = jacobian_matrix[..., 1, :]
        dz = jacobian_matrix[..., 2, :]

        det_a = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        det_b = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        det_c = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])
        determinant = det_a - det_b + det_c

        # return torch.det(jacobian_matrix)
        return determinant




class GradientComputation:
    def __init__(self):
        pass

    def compute_matrix(self, coords, field):
        """
        compute partial derivatives for each dimension of the field wrt each dimension of the coords in parallel.

        Args:
            coords: Tensor of shape [batch_size, num_points, dim].
            field: Tensor of shape [batch_size, num_points, dim].

        Returns:
            matrix: Tensor of shape [batch_size, num_points, dim, dim].
        """
        batch_size, num_points, dim = coords.shape
        # initialize the Jacobian matrix
        matrix = torch.zeros(batch_size, num_points, dim, dim, device=coords.device)

        # loop over dimensions of the field (to calculate gradient w.r.t each dimension)
        for d in range(dim):
            grad_outputs = torch.zeros_like(field)
            grad_outputs[..., d] = 1.0  # set grad_outputs to compute derivative w.r.t. dth output dimension

            # compute gradients for all points accross all batches in parallel
            grad = torch.autograd.grad(outputs=field, inputs=coords, grad_outputs=grad_outputs, create_graph=True, )[0]

            # store the computed gradients in the jac_matrix
            matrix[..., d] = grad
        return matrix

    # def compute_matrix(self, coords, field):
    #     """
    #     Alternative way to do it that is less effitient since i'm looping over each batch and then each dimension
    #     compute partial derivatives for each dimension of the field wrt each dimension of the coords.
    #     useful for computing jacobian matrix

    #     """

    #     batch_size, patch, dim = coords.shape #batch_size, flattened_patchsize, ndims
    #     matrix = torch.zeros(batch_size, patch, dim, dim)

    #     for b in range(batch_size):
    #         for i in range(dim):

    #             matrix[b, :, i, :] = self.gradient(coords, field[b, :,  i], b)

    #     return matrix

    def gradient(self, input_coords, output, b=None, grad_outputs=None):

        grad_outputs = torch.ones_like(output)
        grad = torch.autograd.grad(output, [input_coords], grad_outputs=grad_outputs, create_graph=True)[0]

        if b == None:
            return grad
        else:
            return grad[b]


class MonotonicConstraint:
    """
    Compute the d|J|/dt and penalize non-monotonicity
    """

    def __init__(self, patch_size, batch_size, time, gradient_type):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.time = time
        self.epsilon = 1e-4
        self.gradient_type = gradient_type

    def forward(self, jacobian_determinants):
        jacobian_determinants = jacobian_determinants.view(len(self.time.unique()), self.batch_size, *self.patch_size)
        if self.gradient_type == "analytic_gradient":
            jacobian_determinants = jacobian_determinants.unsqueeze(-1)
            voxelwise_derivatives = torch.autograd.grad(
                outputs=jacobian_determinants,
                inputs=self.time,
                grad_outputs=torch.ones_like(jacobian_determinants),
                create_graph=True,
            )[0]
            voxelwise_derivatives = voxelwise_derivatives.squeeze(-1)
        else:
            ## using finite difference, check this later because of batchsize dimension
            dj = jacobian_determinants[1:] - jacobian_determinants[:-1]
            dt = self.time[1:] - self.time[:-1]
            voxelwise_derivatives = dj / dt
        # mono_loss = torch.min(torch.relu(voxelwise_derivatives - self.epsilon).sum(), torch.relu(-voxelwise_derivatives - self.epsilon).sum())/10000
        mono_loss = torch.min(torch.relu(voxelwise_derivatives).sum(), torch.relu(-voxelwise_derivatives).sum())/10000
        return mono_loss


class FiniteDifference:
    def __init__(self):
        pass

    def compute(self, field):
        """"
        Takes as input field of shape [batch_size, *patch_size, ndims]
        """
        field = field.view(self.batch_size, *self.patch_size, len(self.patch_size))
        spacing = 1  # spacing along x, y and z
        x = field[:, :, :, :, 0]
        y = field[:, :, :, :, 1]
        z = field[:, :, :, :, 2]

        # compute gradients along each axis (x, y, z) for each displacement component, retuns a tuple of gradients along each axis
        gradients_x = torch.gradient(x, dim=(1, 2, 3), spacing=spacing)
        gradients_y = torch.gradient(y, dim=(1, 2, 3), spacing=spacing)
        gradients_z = torch.gradient(z, dim=(1, 2, 3), spacing=spacing)

        # sum of mean squared gradients
        smoothness = sum((grad ** 2).mean() for grad in gradients_x + gradients_y + gradients_z)
        return smoothness

        # du = torch.gradient(field, spacing)
        # dx = du[0]
        # dy = du[1]
        # dz = du[2]
        # return (dx*dx).mean() + (dy*dy).mean() + (dz*dz).mean()


class PenalizeLocalVolumeChange:
    """
    penalizes local volume change
    """

    def __init__(self):
        pass

    def forward(self, coords, field):
        ndims = coords.shape[-1]

        jacobian_matrix = torch.zeros(coords.shape[0], ndims, ndims)

        for i in range(ndims):
            jacobian_matrix[:, i, :] = self.gradient(coords,
                                                     field[:, i])  # derivative of field wrt each dimension in coords
            jacobian_matrix[:, i, i] += torch.ones_like(
                jacobian_matrix[:, i, i])  # add ones on diagonal to preserve local volume across diagnoal elements

        loss = 1 - torch.det(jacobian_matrix)
        loss = torch.mean(
            torch.abs(loss))  # absolute deviation between 1(unity determinant- i.e no volume change) and |J|
        return loss

    def gradient(self, coords, field):
        gradient_outputs = torch.ones_like(field)
        gradient = torch.autograd.grad(field, [coords], grad_outputs=gradient_outputs, create_graph=True)[0]
        return gradient


########################POSITIONAL ENCODING##########################
class PositionalEncoding:
    """
    Each position should be the same size as the output of the time embedding since we are summing them up
    """

    def __init__(self, len_token_index=100, dim_out_embs=64, n=10000):
        self.dim_out_embs = dim_out_embs
        self.n = n
        self.len_token_index = len_token_index  # len_token_index == len_time_object

    def encode(self):
        pos_enc = torch.zeros(self.len_token_index, self.dim_out_embs).to(device)
        for token_index in range(self.len_token_index):
            for i in np.arange(int(self.dim_out_embs / 2)):
                denominator = np.power(self.n, 2 * i / self.dim_out_embs)
                pos_enc[token_index, 2 * i] = np.sin(token_index / denominator)
                pos_enc[token_index, 2 * i + 1] = np.cos(token_index / denominator)
        return pos_enc
