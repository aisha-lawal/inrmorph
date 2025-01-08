import csv
import gc
import itertools
import shutil
import sys
import os
import glob
from typing import List, Tuple
import numpy as np
import torch
import wandb
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)
from utils import SpatialTransform
from data_modules.inrmorph import InrMorphDataModule, define_resolution, get_time_points, CoordsImageTest, load_data
from models.inrmorph import InrMorph
from config import arg, device


def cross_validation_setup():
    # for each hyperparameter combination
    hyperparams = list(itertools.product(spatial, temporal, monotonicity))
    for hyperparam_idx, hyperparam in enumerate(hyperparams):
        spatial_reg, temporal_reg, monotonicity_reg = hyperparam
        print(
            f"Testing hyperparameters: spatial={spatial_reg}, temporal={temporal_reg}, monotonicity={monotonicity_reg}")

        structure_scores = []
        for subject_idx, subjectID in enumerate(subjects):
            # load my data
            subject_datapath = datapath + subjectID + "/resampled/"
            hyperparam_str = "_".join(f"{value}" for value in hyperparam) #make values a string
            logger_name = f"cross_validation_{str(subjectID)}_{hyperparam_str}_subjectidx_{subject_idx}"
            print("subject data path: ", subject_datapath + "I*.nii")
            data = sorted(glob.glob(subject_datapath + "I*.nii"))
            images = define_resolution(data=data, image=True, scale_factor=args.scale_factor)

            print("image shape: ", images[0].shape)
            num_steps_per_epoch = args.num_patches // args.batch_size
            time_points = get_time_points(data)
            time_points = torch.tensor(time_points, device=device, dtype=torch.float32)
            normalised_time_points = time_points / 12
            I0 = images[0]
            It = images

            print("######################Registering {} across time: {} in years##################".format(
                subjectID, time_points))
            patch_size = [args.patch_size for _ in range(len(I0.shape))]

            # create wandb instance
            model_checkpoint = ModelCheckpoint(
                filename="{val_loss:.5f}-{epoch:02d}-" + logger_name,
                save_top_k=1,
                monitor="val_loss",
                mode="min",
            )
            logger = WandbLogger(
                project="inrmorph",
                entity="aishalawal",
                name=logger_name,
                log_model=True,
            )
            # define train/val pipeline
            main_pipeline(
                I0=I0,
                It=It,
                hyperparam=hyperparam,
                patch_size=patch_size,
                logger_name=logger_name,
                logger=logger,
                model_checkpoint=model_checkpoint,
                normalised_time_points=normalised_time_points,
                num_steps_per_epoch=num_steps_per_epoch,
                time_points=time_points,
                subjectID=subjectID,
                subject_datapath=subject_datapath,
                hyperparam_idx=hyperparam_idx,
                subject_idx=subject_idx,
            )


def main_pipeline(
        I0: torch.Tensor,
        It: List,
        hyperparam: Tuple[float, ...],
        patch_size: List[int],
        logger_name: str,
        logger: WandbLogger,
        model_checkpoint: ModelCheckpoint,
        normalised_time_points: torch.Tensor,
        num_steps_per_epoch: int,
        time_points: torch.Tensor,
        subjectID: str,
        subject_datapath: str,
        hyperparam_idx: int,
        subject_idx: int,

):
    spatial_reg, temporal_reg, monotonicity_reg = hyperparam
    data_module = InrMorphDataModule(
        patch_size=patch_size,
        num_patches=args.num_patches,
        val_split=args.val_split,
        batch_size=args.batch_size,
        image=I0
    )

    model = InrMorph(
        I0=I0,
        It=It,
        patch_size=patch_size,
        spatial_reg_weight=spatial_reg,
        temporal_reg_weight=temporal_reg,
        monotonicity_reg_weight=monotonicity_reg,
        spatial_reg_type=args.spatial_reg_type,
        batch_size=args.batch_size,
        network_type=args.network_type,
        similarity_metric=args.similarity_metric,
        gradient_type=args.gradient_type,
        time=normalised_time_points,
        lr=args.lr,
        weight_decay=args.weight_decay,
        omega_0=args.omega_0,
        hidden_layers=args.hidden_layers,
        time_features=args.time_features,
        hidden_features=args.hidden_features,
        num_epochs=args.num_epochs,
    )
    trainer = Trainer(
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.num_epochs,
        log_every_n_steps=num_steps_per_epoch,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=[model_checkpoint],
        logger=logger,
        precision="32")

    # log params to wandb
    model_params = dict(
        patch_size=patch_size,
        num_patches=args.num_patches,
        val_split=args.val_split,
        batch_size=args.batch_size,
        network_type=args.network_type,
        gradient_type=args.gradient_type,
        monotonicity_reg=monotonicity_reg,
        num_epochs=args.num_epochs,
        lr=args.lr,
        hyperparam_idx=hyperparam_idx,
        subject_idx=subject_idx,
        weight_decay=args.weight_decay,
        spatial_reg=spatial_reg,
        temporal_reg=temporal_reg,
        omega_0=args.omega_0,
        hidden_layers=args.hidden_layers,
        subjectID=subjectID,
        time_features=args.time_features,
        hidden_features=args.hidden_features,
        time_points=time_points,
    )
    logger.log_hyperparams(model_params)

    train_generator, val_generator = data_module.dataloaders()

    print("######################Training##################")
    logger.watch(model=model, log_freq=10, log_graph=True)
    trainer.fit(model=model, train_dataloaders=train_generator, val_dataloaders=val_generator)
    # validate on entire image and use to compute the dice and monotonicity
    # use checkpoint

    # pass through model at observed timepoints
    print("##################EVALUATING###############")
    gc.collect()
    torch.cuda.empty_cache()
    evaluation(
        model_checkpoint=model_checkpoint,
        subject_datapath=subject_datapath,
        time_points=time_points,
        normalised_time_points=normalised_time_points,
        I0=I0,
        It=It,
        hyperparam_idx=hyperparam_idx,
        subject_idx=subject_idx,
        spatial_reg=spatial_reg,
        temporal_reg=temporal_reg,
        monotonicity_reg=monotonicity_reg,
        patch_size=patch_size,
    )
    save_logger_name(logger_name)
    wandb.finish()
    shutil.rmtree("wandb")
    shutil.rmtree("logs")


def save_logger_name(logger_name):
    with open('logger_name.txt', 'a') as f:
        f.write(logger_name)
        f.write('\n')
    f.close()


def evaluation(
        model_checkpoint: ModelCheckpoint,
        subject_datapath: str,
        time_points: torch.Tensor,
        normalised_time_points: torch.Tensor,
        I0: torch.Tensor,
        It: List[torch.Tensor],
        hyperparam_idx,
        subject_idx,
        spatial_reg,
        temporal_reg,
        monotonicity_reg,
        patch_size,
):
    # eval generator
    image_vector = CoordsImageTest(I0.shape, scale_factor=1)
    eval_generator = DataLoader(dataset=image_vector, batch_size=211250, shuffle=False)
    print(f"check point path: {model_checkpoint.best_model_path}")
    model = InrMorph.load_from_checkpoint(
        model_checkpoint.best_model_path,
        I0=I0,
        It=It,
        patch_size=patch_size,
        spatial_reg_weight=spatial_reg,
        temporal_reg_weight=temporal_reg,
        monotonicity_reg_weight=monotonicity_reg,
        spatial_reg_type=args.spatial_reg_type,
        batch_size=args.batch_size,
        network_type=args.network_type,
        similarity_metric=args.similarity_metric,
        gradient_type=args.gradient_type,
        time=normalised_time_points,
        lr=args.lr,
        weight_decay=args.weight_decay,
        omega_0=args.omega_0,
        hidden_layers=args.hidden_layers,
        time_features=args.time_features,
        hidden_features=args.hidden_features,
        num_epochs=args.num_epochs,
    )

    model.eval()
    group = subject_datapath.split("/")[6]
    subject_id = subject_datapath.split("/")[7]
    stack_total_deformation_field = []
    stack_total_jac_det = []

    for idx, t in enumerate(time_points):
        tm = normalised_time_points[idx]
        print(f"evaluating index {idx} of time {tm}")
        for k, coords in enumerate(eval_generator):
            coords = coords.squeeze().to(device, dtype=torch.float32).requires_grad_(True)
            displacement_vector = model.test_step(coords, tm).squeeze().to(device)
            deformation_field = torch.add(displacement_vector, coords)
            jac_det = jacobian_determinant(coords, deformation_field)

            coords = coords.cpu().detach()
            deformation_field = deformation_field.cpu().detach()
            jac_det = jac_det.cpu().detach()

            if k == 0:
                total_jac_det = jac_det
                total_deformation_field = deformation_field
            else:
                total_jac_det = torch.cat((total_jac_det, jac_det))
                total_deformation_field = torch.cat((total_deformation_field, deformation_field), 0)

        total_deformation_field = total_deformation_field.view(-1, 3).unsqueeze(0)
        stack_total_deformation_field.append(total_deformation_field)
        stack_total_jac_det.append(total_jac_det.view(I0.shape))

    # load labels to compute dice and as mask for monotonicity
    data_labels = sorted(glob.glob(subject_datapath + "../labels/I*"))
    labels = [load_data(img, False) for img in data_labels]
    I0_seg = labels[0]
    It_seg = labels
    # compute monotonicity
    jac_det_derivative = torch.stack(stack_total_jac_det, axis=0)
    region_mask = I0_seg.cpu().numpy()
    jac_det_derivative = torch.gradient(jac_det_derivative, spacing=(time_points.cpu().detach(),), dim=0)[
        0].cpu().numpy()

    stack_moved_seg = []
    # precompute structure masks
    structure_masks = {
        structure: (region_mask == left_structures[i]) | (region_mask == right_structures[i])
        for i, structure in enumerate(structures)
    }
    structure_dice_all_timepoints = []
    structure_sign_consistency = {structure: [] for structure in structures}

    for time_idx, selected_time in enumerate(time_points):
        # dice
        img_seg = It_seg[time_idx]
        moved_seg = transform.nearest_neighbor_interpolation(
            stack_total_deformation_field[time_idx].to(device),
            img_seg
        ).view(img_seg.shape)
        stack_moved_seg.append(moved_seg.cpu().numpy().squeeze())
        structure_dice = combine_labels(I0_seg.cpu().numpy(), stack_moved_seg[time_idx])
        structure_dice_all_timepoints.append(structure_dice)

        # compute monotonicity
        for structure, mask in structure_masks.items():
            avg_consistency = 0
            if np.any(mask):
                structure_voxel_signs = np.sign(jac_det_derivative[time_idx][mask])
                avg_consistency = np.mean(structure_voxel_signs == structure_voxel_signs[0])
            structure_sign_consistency[structure].append(avg_consistency)

    # total proportion
    total_proportion_per_structure = {}
    for structure, mask in structure_masks.items():
        if np.any(mask):
            structure_voxels = jac_det_derivative[:, mask]
            voxel_signs = np.sign(structure_voxels)
            voxel_consistency = np.mean(voxel_signs == voxel_signs[0, :], axis=0)
            total_proportion_per_structure[structure] = voxel_consistency.mean()


    wandb_table = wandb.Table(columns=csv_columns) #to of wand b
    with open(result_path, 'a') as f:
        writer = csv.writer(f)
        for time_idx, selected_time in enumerate(time_points):
            for structure in structures:
                dice = structure_dice_all_timepoints[time_idx][structure]
                avg_consistency = structure_sign_consistency[structure][time_idx]
                total_proportion = total_proportion_per_structure[structure]

                jacobian_metrics = compute_jacobian_metrics(
                    stack_total_jac_det[time_idx], structure_masks[structure]
                )
                derivative_metrics = compute_derivative_metrics(
                    jac_det_derivative[time_idx], structure_masks[structure]
                )

                row = [
                    subject_id, hyperparam_idx, subject_idx, spatial_reg, temporal_reg, monotonicity_reg,
                    selected_time.item(), structure, dice, avg_consistency, total_proportion,
                    *jacobian_metrics,
                    *derivative_metrics,
                    group
                ]
                writer.writerow(row)
                wandb_table.add_data(*row)
            wandb.log({"Cross val metrics": wandb_table})
    #free up memory
    del stack_total_deformation_field
    del stack_total_jac_det
    gc.collect()
    torch.cuda.empty_cache()

def dice_score(label1, label2):
    intersection = np.sum(label1[label1 > 0] == label2[label1 > 0])
    dice = (2 * intersection) / (np.sum(label1) + np.sum(label2))
    return dice

def combine_labels(img1, img2):
    structure_dice = {}
    for i, structure in enumerate(structures):
        img1_combined = np.where(np.isin(img1, [left_structures[i], right_structures[i]]), 1, 0)
        img2_combined = np.where(np.isin(img2, [left_structures[i], right_structures[i]]), 1, 0)
        dice = dice_score(img1_combined, img2_combined)
        structure_dice[structure] = dice
    return structure_dice


# computing jacobian metrics
def compute_jacobian_metrics(det, mask):
    jacobian_values = det[mask > 0]
    jac_det_mean = jacobian_values.mean().item()
    jac_det_min = jacobian_values.min().item()
    jac_det_max = jacobian_values.max().item()
    number_of_contractions = ((jacobian_values > 0) & (jacobian_values < 1)).sum().item()
    no_volume_change = (jacobian_values == 1).sum().item()
    folded_voxels = (jacobian_values < 0).sum().item()
    return jac_det_mean, jac_det_min, jac_det_max, number_of_contractions, no_volume_change, folded_voxels


def compute_derivative_metrics(jac_det_derivative, mask):
    derivative_values = jac_det_derivative[mask > 0]
    jac_det_derivative_mean = derivative_values.mean().item()
    jac_det_derivative_min = derivative_values.min().item()
    jac_det_derivative_max = derivative_values.max().item()
    return jac_det_derivative_mean, jac_det_derivative_min, jac_det_derivative_max


def jacobian_determinant(coords, deformation_field):
    jac = compute_jacobian_matrix(coords, deformation_field)
    return torch.det(jac)


def compute_jacobian_matrix(coords, deformation_field):
    dim = coords.shape[1]
    jacobian_matrix = torch.zeros(coords.shape[0], dim, dim)

    for i in range(dim):
        jacobian_matrix[:, i, :] = gradient(coords, deformation_field[:, i])
    return jacobian_matrix


def gradient(coords, output, grad_outputs=None):
    grad_outputs = torch.ones_like(output)
    grad = torch.autograd.grad(output, [coords], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


if __name__ == "__main__":
    args = arg()
    datapath = "/srv/thetis2/as2614/inrmorph/data/"
    structures = [
        "Cerebral White Matter",
        "Cerebral Cortex",
        "Lateral Ventricle",
        "Inferior Lateral Ventricle",
        "Cerebellum White Matter",
        "Cerebellum Cortex",
        "Thalamus",
        "Caudate",
        "Putamen",
        "Pallidum",
        "Hippocampus",
        "Amygdala",
        "Accumbens Area",
        "Ventral DC"
    ]
    left_structures = [2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28]
    right_structures = [41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]
    paired_structures = {
        structure: (left, right)
        for structure, left, right in zip(structures, left_structures, right_structures)
    }
    # create result csv with structure fold_index
    result_path = "cross_val_results.csv"
    csv_columns =[ "subjectID", "hyperparam_idx", "subject_idx", "spatial_reg", "temporal_reg", "monotonicity_reg",
                "time_point", "structure", "structure_dice", "structure_proportion_monotonicity",
                "total_proportion_monotonicity", "jac_det_mean",
                "jac_det_min", "jac_det_max", "number_of_contractions", "no_volume_change", "folded_voxels",
                "jac_det_derivative_mean", "jac_det_derivative_min", "jac_det_derivative_max", "group"
            ]
    if not os.path.exists(result_path):
        with open(result_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(csv_columns)

    transform = SpatialTransform()
    spatial = [0.01, 0.001, 0.1]
    temporal = [0.1, 0.5, 1.0]
    monotonicity = [0.1, 0.5, 1.0]

    subjects = ["AD/005_S_0814", "MCI/099_S_4076", "CN/002_S_0413", "MCI/041_S_0679", "AD/007_S_4637", "CN/136_S_4269"]
    cross_validation_setup()
