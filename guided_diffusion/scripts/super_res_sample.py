"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import matplotlib.pyplot as plt
import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
import statistics
from torch.utils.data import DataLoader
from guided_diffusion.image_datasets import load_data
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    get_psnr,
    get_ssim,
    args_to_dict,
    add_dict_to_argparser,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    #
    # data = load_data_for_worker(args.base_samples, args.batch_size, args.class_cond)

    data = load_superres_data(
        args.data_dir,
        args.hr_data_dir,
        args.lr_data_dir,
        args.other_data_dir,
        args.batch_size,
    )

    logger.log("creating samples...")

    psnr_list, ssim_list = [], []
    all_images = []

    while len(all_images) * args.batch_size < args.num_samples:
        hr, model_kwargs = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 1, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )

        hr = hr.permute(0, 2, 3, 1)
        hr = hr.contiguous()
        hr = hr.cpu().numpy()

        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()

        model_kwargs['low_res'] = model_kwargs['low_res'].permute(0, 2, 3, 1)
        model_kwargs['low_res'] = model_kwargs['low_res'].contiguous()
        model_kwargs['low_res'] = model_kwargs['low_res'].cpu().numpy()

        for i in range(hr.shape[0]):
            psnr_list.append(get_psnr(hr[i, ...], sample[i, ...]))
            ssim_list.append(get_ssim(hr[i, ...], sample[i, ...]))

        for j in range(len(psnr_list)):
            f, axs = plt.subplots(1, 3, figsize=(15, 15))
            axs[0].imshow(hr[j, ..., 0].T, cmap='gray')
            axs[0].set_title('Ground Truth')
            axs[1].imshow(sample[j, ..., 0].T, cmap='gray')
            axs[1].set_title('Generated HR image')
            axs[2].imshow(model_kwargs['low_res'][j, ..., 0].T, cmap='gray')
            axs[2].set_title('LR image')
            plt.show()

        print()
        print(f'Number of evaluated slices: {len(psnr_list)}')
        print(f'Mean PSNR: {statistics.mean(psnr_list)}')
        print(f'Mean SSIM: {statistics.mean(ssim_list)}')

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def load_superres_data(data_dir, hr_data_dir, lr_data_dir, other_data_dir, batch_size, class_cond=False):
    dataset = load_data(
        hr_data_dir=hr_data_dir,
        lr_data_dir=lr_data_dir,
        other_data_dir=other_data_dir)
    
    loader = DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

    for _, (hr_data, lr_data, other_data) in enumerate(loader):
        model_kwargs = {"low_res": lr_data, "other": other_data}
        yield hr_data, model_kwargs


def load_data_for_worker(base_samples, batch_size, class_cond):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if class_cond:
            label_arr = obj["arr_1"]
    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    while True:
        for i in range(rank, len(image_arr), num_ranks):
            buffer.append(image_arr[i])
            if class_cond:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = th.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                buffer, label_buffer = [], []


def create_argparser():
    defaults = dict(
        data_dir='',
        hr_data_dir="new_dataset/IXI_testing_hr_t2_scale_by_4_imgs.npy",
        lr_data_dir="new_dataset/IXI_testing_lr_t2_scale_by_4_imgs.npy",
        other_data_dir="new_dataset/IXI_testing_hr_t1_scale_by_4_imgs.npy",
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        base_samples="",
        model_path="models/final_ema_16.pt",
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
