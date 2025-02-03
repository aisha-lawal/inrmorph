import glob
import torch
from lightning import Trainer
from config import save_logger_name, arg, wandb_setup, device
from data_modules.inrmorph import InrMorphDataModule, define_resolution, get_time_points, save_noisy_data
from models.inrmorph import InrMorph
import os


def main():
    data_module = InrMorphDataModule(
        patch_size=patch_size,
        num_patches=args.num_patches,
        val_split=args.val_split,
        batch_size=args.batch_size,
        image=I0
    )
    train_generator, val_generator = data_module.dataloaders()
    model = InrMorph(
        I0=I0,
        It=It,
        patch_size=patch_size,
        spatial_reg_weight=args.spatial_reg,
        temporal_reg_weight=args.temporal_reg,
        monotonicity_reg_weight=args.monotonicity_reg,
        spatial_reg_type=args.spatial_reg_type,
        batch_size=args.batch_size,
        network_type=args.network_type,
        similarity_metric=args.similarity_metric,
        gradient_type=args.gradient_type,
        time=normalised_time_points,
        observed_time_points=observed_time_points,
        lr=args.lr,
        weight_decay=args.weight_decay,
        omega_0=args.omega_0,
        hidden_layers=args.hidden_layers,
        time_features=args.time_features,
        hidden_features=args.hidden_features,
        num_epochs=args.num_epochs,
        extrapolate=args.extrapolate,
        l2_weight=args.l2_weight,
        fixed_time_embedding=args.fixed_time_embedding
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
        subjectID=args.subjectID,
        time_features=args.time_features,
        hidden_features=args.hidden_features,
        time_points=time_points,
        add_noise=args.add_noise,
        noise_mean=args.noise_mean,
        noise_std=args.noise_std,
        extrapolate=args.extrapolate,
        interpolate=args.interpolate,
        l2_weight=args.l2_weight,
        fixed_time_embedding=args.fixed_time_embedding
    )
    logger.log_hyperparams(model_params)

    print("######################Training##################")
    logger.watch(model=model, log_freq=10, log_graph=True)
    trainer.fit(model=model, train_dataloaders=train_generator, val_dataloaders=val_generator)
    save_logger_name(args.logger_name)


if __name__ == "__main__":
    args = arg()
    model_checkpoint, logger = wandb_setup()
    """
    Generating data resolution
    """
    datapath = args.datapath + args.subjectID + "/resampled/"
    print("data path: ", datapath)
    data = sorted(glob.glob(datapath + "I*.nii"))
    images = define_resolution(data=data, image=True, add_noise=args.add_noise, noise_mean=args.noise_mean, noise_std=args.noise_std, scale_factor=args.scale_factor)

    """
    Retrieve time points
    """
    print("image shape: ", images[0].shape)
    num_steps_per_epoch = args.num_patches // args.batch_size
    I0 = images[0]  
    It = images 

    #save images if add_noise is True to use for eval later
    if args.add_noise:
        output_dir = f"eval/noisy_data_l2_100_rand/std_{args.noise_std}_{args.subjectID.replace('/', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        save_noisy_data(data, images, output_dir)


    time_points = get_time_points(data)
    
    time_points = torch.tensor(time_points, device=device, dtype=torch.float32)
    observed_time_points = (time_points / 12).tolist()
    #extrapolating last time point, 6 and 12 months after last time point
    if args.extrapolate: 
        #exclude last time point so we compute the dice at eval time
        observed_time_points = (time_points[:-1] / 12).tolist()
        It=It[:-1]
        # or just add more displacement fields to the last time point to extrapolate
        # time_points = torch.cat((time_points, torch.tensor([time_points[-1] + 12], device=device,  dtype=torch.float32)))

    if args.interpolate: 
        observed_time_points = [time_points[0], time_points[1], time_points[-1]]
        observed_time_points = [otp / 12 for otp in observed_time_points]
        It=[It[0], It[1], It[-1]]
        #to add to exixting observed time points
        # observed_time_points = [time_points[0], time_points[1], time_points[2], time_points[-1]]
        # observed_time_points = [otp / 12 for otp in observed_time_points]
        # It=[It[0], It[1], It[2], It[-1]]
        
    normalised_time_points = time_points / 12


    print("######################Registering {} across time: {} in years##################".format(datapath, time_points))
    patch_size = [args.patch_size for _ in range(len(I0.shape))]
    main()
