import glob
from lightning import Trainer
from torch.utils.data import DataLoader
from config import *
from data_modules.inrmorph import CoordsPatch, InrMorphDataModule, define_resolution
from models.inrmorph import InrMorph
from data_modules.inrmorph import load_data


def main():
    data_module = InrMorphDataModule(
        patch_size=patch_size,
        num_patches=args.num_patches,
        val_split=args.val_split,
        batch_size=args.batch_size,
        image_shape=I0.shape
    )
    train_generator, val_generator = data_module.dataloaders()
    model = InrMorph(I0, It, patch_size, spatial_reg, temporal_reg, monotonicity_reg, args.batch_size,
                     args.network_type,
                     args.similarity_metric, args.gradient_type, args.loss_type, normalised_time_points)

    trainer = Trainer(fast_dev_run=args.fast_dev_run, max_epochs=args.num_epochs,
                      log_every_n_steps=num_steps_per_epoch // 2, accelerator="auto",
                      devices="auto",
                      strategy="auto", callbacks=[model_checkpoint], logger=logger, precision="32")

    print("######################Training##################")
    logger.watch(model=model, log_freq=10, log_graph=True)
    trainer.fit(model=model, train_dataloaders=train_generator, val_dataloaders=val_generator)

    print(f"Allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / (1024 * 1024)} MB")
    save_logger_name(args.logger_name)


if __name__ == "__main__":
    args = arg()
    model_checkpoint, _, logger = wandb_setup()
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
    # spatial_reg = 0.01 #for siren
    spatial_reg = 1
    monotonicity_reg = 0.15
    temporal_reg = 0.001
    main()
