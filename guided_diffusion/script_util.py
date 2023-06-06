import argparse
from matplotlib import pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import SuperResModel

def draw_sample_image(x, postfix):
    """
    Draw and display a sample image.
    Args:
        x (torch.Tensor): Input image tensor.
        postfix (str): Additional text for the title of the image.
    """
    plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.show()

def get_ssim(ground_truth, generated_image):
    """
    Calculate the structural similarity index (SSIM) between two images.
    Args:
        ground_truth (numpy.ndarray): Ground truth image.
        generated_image (numpy.ndarray): Generated image.
    Returns:
        float: SSIM value.
    """
    return structural_similarity(ground_truth, generated_image, multichannel=True, gaussian_weights=True, sigma=1.5)

def get_psnr(ground_truth, generated_image):
    """
    Calculate the peak signal-to-noise ratio (PSNR) between two images.
    Args:
        ground_truth (numpy.ndarray): Ground truth image.
        generated_image (numpy.ndarray): Generated image.
    Returns:
        float: PSNR value.
    """
    return peak_signal_noise_ratio(ground_truth, generated_image)

def sr_create_model_and_diffusion(args):
    """
    Create a super-resolution model and diffusion instance.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    Returns:
        tuple: Tuple containing the model and diffusion instances.
    """
    model = sr_create_model(
        args.image_size,
        args.in_channel,
        args.num_channels,
        args.num_res_blocks,
        learn_sigma=args.learn_sigma,
        use_checkpoint=args.use_checkpoint,
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
    )
    diffusion = create_gaussian_diffusion(
        steps=args.diffusion_steps,
        learn_sigma=args.learn_sigma,
        noise_schedule=args.noise_schedule,
        use_kl=args.use_kl,
        predict_xstart=args.predict_xstart,
        rescale_timesteps=args.rescale_timesteps,
        rescale_learned_sigmas=args.rescale_learned_sigmas,
        timestep_respacing=args.timestep_respacing,
    )
    return model, diffusion

def sr_create_model(
        image_size,
        in_channel,
        num_channels,
        num_res_blocks,
        learn_sigma,
        use_checkpoint,
        attention_resolutions,
        num_heads,
        num_head_channels,
        num_heads_upsample,
        use_scale_shift_norm,
        dropout,
        resblock_updown,
        use_fp16,
):
    channel_mult = (1, 1, 2, 2, 3, 3)

    attention_ds = [8, 16, 32]

    return SuperResModel(
        image_size=image_size,
        in_channels=in_channel,
        model_channels=num_channels,
        out_channels=(in_channel if not learn_sigma else in_channel * 2),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
    )


def create_gaussian_diffusion(
        *,
        steps=1000,
        learn_sigma=False,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
