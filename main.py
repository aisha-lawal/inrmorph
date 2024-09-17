from settings import *
from utils import generate_random_sum, generate_time_sequence, load_data, CoordsPatchesTrain, CoordsImageTest, CoordsPatch
from inrmorph import INRMorph


def main():
    train_generator = DataLoader(dataset=CoordsPatch(patch_size=patch_size,
                                                     npatches=npatches_train, imgshape=I_0.shape),
                                 batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True)
    val_generator = DataLoader(dataset=CoordsPatch(patch_size=patch_size,
                                                   npatches=npatches_val, imgshape=I_0.shape), batch_size=batch_size,
                               shuffle=False, num_workers=8, drop_last=True)

    model = INRMorph(I_0, I_t, patch_size, spatial_reg, temporal_reg, batch_size, args.network_type,
                            args.similarity_metric, args.gradient_type, args.loss_type, time_seq, observed_indx)

    # model = torch.compile(model)
    trainer = Trainer(fast_dev_run=False, max_epochs=150, log_every_n_steps=50, accelerator="auto", devices="auto",
                      strategy="auto", callbacks=[model_checkpoint], logger=logger, precision="32")
    # trainer = Trainer(fast_dev_run=True, max_epochs=50, log_every_n_steps=150, accelerator="auto", devices="auto",strategy="auto", precision="32", overfit_batches=100)
    trainer.fit(model=model, train_dataloaders=train_generator, val_dataloaders=val_generator)

    print(f"Allocated: {torch.cuda.memory_allocated() / (1024 * 1024)} MB")
    print(f"Cached: {torch.cuda.memory_reserved() / (1024 * 1024)} MB")
    save_artifacts(trainer, model_checkpoint, args.logger_name, logger)


if __name__ == "__main__":

    args = arg()
    model_checkpoint, early_stopping, logger = wandb_setup()
    datapath = "dataset/" + args.subjectID + "/resampled/"
    data = sorted(glob.glob(datapath + "I*.nii"))

    images = []
    for img in data:
        images.append(load_data(img))

    time_points = torch.tensor(args.time)
    normalised_time_points = time_points/12
    time_seq_spacing = [35, 15, 52]
    observed_indx = [34, 48, 99]
    time_seq = generate_time_sequence(normalised_time_points, time_seq_spacing)
    time_seq = torch.tensor(time_seq, device=device, dtype=torch.float32)
    I_0 = images[0]  # moving #260, 260, 200
    I_t = images[2]
    print("######################Registering across time: {} in years##################".format(args.time))

    patch_size = [24 for _ in range(len(I_0.shape))]

    npatches_train = 900
    npatches_val = 300
    batch_size = 4
    spatial_reg = 1e+4
    temporal_reg = 1e+6
    main()
