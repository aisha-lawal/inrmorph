import glob
from datetime import datetime
import random
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import functional as F
import nibabel as nib
from config import device, set_seed


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
        self.set_seed = set_seed(42)

        self.patch_size = np.ceil(np.array(patch_size) / 2).astype(np.int16)
        patch_dx_dims = torch.tensor(self.patch_size) * self.dx

        patch_coords = [torch.linspace(-patch_dx_dims[i], patch_dx_dims[i],
                                       2 * self.patch_size[i]) for i in range(self.ndims)]

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


class InrMorphDataModule:
    def __init__(self,
                 patch_size: List[int],
                 num_patches: int,
                 val_split: float,
                 batch_size: int,
                 image_shape: torch.Size):
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.val_split = val_split
        self.batch_size = batch_size
        self.image_shape = image_shape

        self.seed = 42
        self.num_workers = 8
        self.drop_last = True

    def generator(self):
        return torch.Generator().manual_seed(self.seed)

    # take all num_patches then divide after
    def dataloaders(self):
        dataset = CoordsPatch(patch_size=self.patch_size,
                              num_patches=self.num_patches, img_shape=self.image_shape)
        # dataset above returns total number of patches
        val_size = int(self.val_split * len(dataset))
        train_size = len(dataset) - val_size

        # split patches based on val porportion
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=self.generator())
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, drop_last=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True,
                                num_workers=self.num_workers, drop_last=True)
        print("train is ", len(train_loader), "val is", len(val_loader))
        return train_loader, val_loader


class CoordsImageTest(Dataset):
    def __init__(self, img_shape, scale_factor=1):
        """
        input: coords with shape (img_height, img_width, ndims)
        Returns: flatten the image where each point has ndims, returns shape (img_shape, ndims)
        """
        # should return different coordinates for training and validation
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


def define_resolution(data, image: bool, scale_factor):
    data = [load_data(img, image) for img in data]

    # F.interpolate has to be of batch, channel, D, H, W, so we stack below
    if scale_factor != 1:
        mode = 'trilinear' if image else 'nearest'
        align_corners = True if mode == 'trilinear' else None
        data = torch.stack(data).unsqueeze(0).permute(1, 0, 2, 3, 4)
        data = F.interpolate(data, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

        # unstack and return list
        data = [t.squeeze(0).squeeze(0) for t in torch.split(tensor=data, split_size_or_sections=1, dim=0)]
        return data
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


if __name__ == "__main__":
    I0 = torch.rand(260, 260, 200)
    num_patches_train = 2000
    num_patches_val = 1000
    patch_size = [12 for _ in range(len(I0.shape))]
    batch_size = 48
    data_module = InrMorphDataModule(
        patch_size=[12, 12, 12],
        num_patches=2000,
        val_split=0.3,
        batch_size=1,
        image_shape=I0.shape
    )
    data_module.dataloaders()

    # data = torch.rand(260, 260, 200)
    datapath = "/srv/thetis2/as2614/inrmorph/data/AD/005_S_0814/resampled/"
    data = sorted(glob.glob(datapath + "I*.nii"))
    datamask = sorted(glob.glob(datapath + "/masks/I*.nii.gz"))
    data = define_resolution(data, image=True, scale_factor=0.5)
    print(len(data), data[0].shape)

    train_generator = DataLoader(dataset=CoordsPatch(patch_size=patch_size,
                                                     num_patches=num_patches_train, img_shape=I0.shape),
                                 batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_generator = DataLoader(dataset=CoordsPatch(patch_size=patch_size,
                                                   num_patches=num_patches_val, img_shape=I0.shape),
                               batch_size=batch_size,
                               shuffle=False, num_workers=8, drop_last=True)
    print("train is ", len(train_generator), "val is", len(val_generator))
    # print first data in train_generator
    print(next(iter(train_generator))[0].shape)
#
