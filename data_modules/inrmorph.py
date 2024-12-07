from datetime import datetime
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import nibabel as nib
from config import device


# training is patchwise, val/test is full image!
class CoordsPatchesTrain(Dataset):
    def __init__(self, patch_size, num_patches, img_shape):
        """doesnt account for non factors, look into later"""

        self.patch_size = patch_size
        assert len(self.patch_size) == 3, "incorrect patchsize, working with 3D data"

        self.img_shape = img_shape
        self.coords = define_coords(self.img_shape)
        self.num_patches = num_patches

        # adjust for ndims
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
        return self.num_patches

    def __getitem__(self, idx):
        # randomly sample num_patches(with replacement)
        inds = torch.randint(0, len(self.coords), (self.num_patches,))
        self.coords = self.coords[inds]  # num_patches, *patch_hwd, dims
        return self.coords[idx]


class CoordsPatch(Dataset):
    def __init__(self, patch_size, num_patches, img_shape):
        # super(self, CoordsPatch).__init__()
        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        self.ndims = len(self.patch_size)
        self.img_shape = img_shape
        self.coords = define_coords(self.img_shape)
        self.dx = torch.div(2, torch.tensor(self.coords.shape[:-1]))
        self.num_patches = num_patches  # how many random patches to sample

        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        patch_dx_dims = torch.tensor(self.patch_size) * self.dx

        patch_coords = [torch.linspace(-patch_dx_dims[i], patch_dx_dims[i], 2 * self.patch_size[i]) for i in
                        range(self.ndims)]
        patch_coords = torch.meshgrid(*patch_coords, indexing=None)
        self.patch_coords = torch.stack(patch_coords, dim=self.ndims)

        coords = self.coords[self.patch_size[0]:-self.patch_size[0],
                 self.patch_size[1]:-self.patch_size[1], self.patch_size[2]:-self.patch_size[2], ...]

        self.spatial_size = coords.shape[:-1]
        self.coords = coords

    def __len__(self):
        return self.num_patches

    def __getitem__(self, idx):
        indx = np.random.randint(0, np.prod(self.spatial_size))

        inds = np.unravel_index(indx, self.spatial_size)

        center = self.coords[inds[0], inds[1], inds[2], :]
        coords = torch.clone(self.patch_coords)

        coords[..., 0] = coords[..., 0] + center[0]
        coords[..., 1] = coords[..., 1] + center[1]
        coords[..., 2] = coords[..., 2] + center[2]
        return coords


class CoordsImageTest(Dataset):
    def __init__(self, img_shape, scale_factor=1):
        """
        input: coords with shape (img_height, img_width, ndims)
        Returns: flatten the image where each point has ndims, returns shape (img_shape, ndims)
        """
        self.img_shape = img_shape
        self.coords = define_coords(self.img_shape)
        self.scale_factor = scale_factor

        if self.scale_factor != 1:  # to test with larger resolution
            self.coords = F.interpolate(self.coords.permute(3, 2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor,
                                        mode='trilinear', align_corners=True)
            self.coords = self.coords.squeeze().permute(2, 3, 1, 0)

            # coords = F.interpolate(coords.permute(2, 0, 1).unsqueeze(0), scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
            # coords = coords.squeeze().permute(1, 2, 0)

        self.ndims = self.coords.shape[-1]
        self.coords = self.coords.view([np.prod(self.coords.shape[:-1]), self.ndims])

    def __len__(self):
        return self.coords.shape[0]  # flattened img_shape

    def __getitem__(self, idx):
        return self.coords[idx, :]



def define_coords(img_shape) -> torch.Tensor:
    """
    defines coordinate between -1 to 1 of shape ndims
    returns tensor of shape (*img_shape, ndims)
    """
    ndims = len(img_shape)
    coords = [torch.linspace(-1, 1, img_shape[i])
              for i in range(ndims)]

    coords = torch.meshgrid(*coords, indexing=None)
    coords = torch.stack(coords, dim=ndims)

    return coords



################DATA LOADING AND ###########

def load_data(path: str, image) -> torch.Tensor:  # 260, 260, 200
    data = np.array(nib.load(path).get_fdata())
    data = torch.tensor(data, device=device, dtype=torch.float32)
    if image:
        # normalize for only images, not labels/masks
        return normalise(data)
    return data


def normalise(img: torch.Tensor) -> torch.Tensor:
    img = (img - img.min()) / (img.max() - img.min())

    return (2 * img) - 1


def get_time_points(data):
    """
    Gets data path and returns time points between each image and the first image in months.
    """
    splitpath = [data[i].split("/")[-1] for i in range(len(data))]
    dates = [datetime.strptime(img.split('_')[1] + '_' + img.split('_')[2], "%Y_%m") for img in splitpath]
    first_date = dates[0]
    time_points = [0] + [(date.year - first_date.year) * 12 + (date.month - first_date.month) for date in dates[1:]]
    return time_points

    #########################TIME SEQUENCE GENERATION#########################


def generate_random_sum(len_time_seq, n):
    cut_points = sorted(random.sample(range(len_time_seq), n - 1))
    values = [cut_points[0]] + [cut_points[i] - cut_points[i - 1] for i in range(1, n - 1)] + [
        len_time_seq - cut_points[-1]]
    return values


def generate_time_sequence(time_points, random_indexes):
    time_seq = []
    for i in range(len(time_points) - 1):
        random_values = np.random.uniform(time_points[i],
                                          time_points[i + 1], random_indexes[i] - 2)
        random_values = np.concatenate(([time_points[i]],
                                        np.sort(random_values), [time_points[i + 1]]))
        time_seq = np.concatenate((time_seq, random_values))
    time_seq = np.unique(time_seq)
    return time_seq

