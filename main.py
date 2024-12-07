import glob
from lightning import Trainer
from torch.utils.data import DataLoader
from config import *
from data_modules.inrmorph import CoordsPatch
from models.inrmorph import InrMorph
from data_modules.inrmorph import load_data


def main():
    train_generator = DataLoader(dataset=CoordsPatch(patch_size=patch_size,
                                                     num_patches=num_patches_train, img_shape=I0.shape),
                                 batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_generator = DataLoader(dataset=CoordsPatch(patch_size=patch_size,
                                                   num_patches=num_patches_val, img_shape=I0.shape),
                               batch_size=batch_size,
                               shuffle=False, num_workers=8, drop_last=True)
    model = InrMorph(I0, It, patch_size, spatial_reg, temporal_reg, monotonicity_reg, batch_size, args.network_type,
                     args.similarity_metric, args.gradient_type, args.loss_type, normalised_time_points)

    trainer = Trainer(fast_dev_run=args.fast_dev_run, max_epochs=250, log_every_n_steps=50, accelerator="auto",
                      devices="auto",
                      strategy="auto", callbacks=[model_checkpoint], logger=logger, precision="32")

    print("######################Training##################")
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

    images = [load_data(img, True) for img in data]
    masks = [load_data(mask, False) for mask in datamask]
    """
    Retrieve time points
    """
    print("image shape: ", images[0].shape)

    # time_points = get_time_points(data)
    print(len(images), len(masks))

    time_points = torch.tensor(args.time, device=device, dtype=torch.float32)
    normalised_time_points = time_points / 12
    I0 = images[0]  # moving #260, 260, 200
    It = images

    I0_mask = masks[0]
    It_mask = masks

    print("######################Registering across time: {} in years##################".format(args.time))

    patch_size = [12 for _ in range(len(I0.shape))]

    num_patches_train = 2000
    num_patches_val = 1000
    # num_patches_train = 1200
    # num_patches_val = 600
    batch_size = 48
    # spatial_reg = 0.01 #for siren
    spatial_reg = 100
    monotonicity_reg = 0.15
    temporal_reg = 1e+6
    main()
