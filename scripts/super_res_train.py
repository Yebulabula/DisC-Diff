"""
Train a super-resolution model.
"""

import argparse

import numpy as np
import torch.nn.functional as F
from skimage.measure import shannon_entropy
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser
)
from guided_diffusion.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure(dir = "models/x4_IXI/")

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    
    # model.load_state_dict(
    #     dist_util.load_state_dict("/home/cbtil/Documents/SRDIFF/guided-diffusion/models/x4_IXI/ema_0.9999_010000.pt", map_location="cpu")
    # )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    brain_dataset = load_superres_data(
        args.hr_data_dir,
        args.lr_data_dir,
        args.other_data_dir
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=brain_dataset,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_superres_data(hr_data_dir, lr_data_dir, other_data_dir):
    return load_data(
        hr_data_dir=hr_data_dir,
        lr_data_dir=lr_data_dir,
        other_data_dir=other_data_dir)


def create_argparser():
    defaults = dict(
        data_dir="",
        hr_data_dir="new_dataset/IXI_training_hr_t2_scale_by_4_imgs.npy",
        lr_data_dir="new_dataset/IXI_training_lr_t2_scale_by_4_imgs.npy",
        other_data_dir="new_dataset/IXI_training_hr_t1_scale_by_4_imgs.npy",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
