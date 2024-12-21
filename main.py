import glob
import torch
from lightning import Trainer
from config import save_logger_name, arg, wandb_setup, device
from data_modules.inrmorph import InrMorphDataModule, define_resolution
from models.inrmorph import InrMorph


def main():
    data_module = InrMorphDataModule(
        patch_size=patch_size,
        num_patches=args.num_patches,
        val_split=args.val_split,
        batch_size=args.batch_size,
        image_shape=I0.shape
    )
    train_generator, val_generator = data_module.dataloaders()
    model = InrMorph(
        I0=I0,
        It=It,
        patch_size=patch_size,
        spatial_reg_weight=args.spatial_reg,
        temporal_reg_weight=args.temporal_reg,
        monotonicity_reg_weight=args.monotonicity_reg,
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
    )
    trainer = Trainer(
        # fast_dev_run=args.fast_dev_run,
        fast_dev_run=True,
        max_epochs=args.num_epochs,
        log_every_n_steps=num_steps_per_epoch,
        accelerator="auto",
        devices="auto",
        strategy="auto",
        callbacks=[model_checkpoint],
        logger=logger,
        precision="32")

    #log params to wandb
    model_params = dict(
        patch_size=patch_size,
        num_patches=args.num_patches,
        val_split=args.val_split,
        batch_size=args.batch_size,
        network_type=args.network_type,
        gradient_type=args.gradient_type,
        monotonicity_reg=args.monotonicity_reg,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        spatial_reg=args.spatial_reg,
        temporal_reg=args.temporal_reg,
        omega_0=args.omega_0,
        hidden_layers=args.hidden_layers,
        time_features=args.time_features,
        hidden_features=args.hidden_features,
    )
    # logger.log_hyperparams(model_params)

    print("######################Training##################")
    # logger.watch(model=model, log_freq=10, log_graph=True)
    trainer.fit(model=model, train_dataloaders=train_generator, val_dataloaders=val_generator)
    save_logger_name(args.logger_name)


if __name__ == "__main__":
    args = arg()
    model_checkpoint, logger = wandb_setup()
    """
    Generating data resolution
    """
    print("data path: ", args.datapath)
    datapath = args.datapath + args.subjectID + "/resampled/"
    data = sorted(glob.glob(datapath + "I*.nii"))
    datamask = sorted(glob.glob(datapath + "/masks/I*.nii.gz"))
    images = define_resolution(data=data, image=True, scale_factor=args.scale_factor)
    masks = define_resolution(data=datamask, image=False, scale_factor=args.scale_factor)

    """
    Retrieve time points
    """
    print("image shape: ", images[0].shape, masks[0].shape)
    num_steps_per_epoch = args.num_patches // args.batch_size
    # time_points = get_time_points(data)
    time_points = torch.tensor(args.time, device=device, dtype=torch.float32)
    normalised_time_points = time_points / 12
    I0 = images[0]  # moving #260, 260, 200
    It = images
    I0_mask = masks[0]
    It_mask = masks

    print("######################Registering across time: {} in years##################".format(args.time))
    patch_size = [args.patch_size for _ in range(len(I0.shape))]
    main()
