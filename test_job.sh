#!/bin/bash
export PYTHONPATH="/home/cbtil/Documents/SRDIFF/guided-diffusion"
MODEL_FLAGS="--attention_resolutions 32,16,8 --diffusion_steps 1000 --image_size 224 --in_channel 1 --learn_sigma True --noise_schedule linear --num_channels 96 --num_head_channels 48 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-4 --batch_size 5"
SAMPLE_FLAGS="--batch_size 10 --num_samples 50000 --timestep_respacing 100 --use_ddim False"
python scripts/super_res_sample.py $MODEL_FLAGS $SAMPLE_FLAGS