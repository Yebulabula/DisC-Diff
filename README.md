# DisC-Diff
This is the PyTorch implementation of Disc-Diff for DisC-Diff: Disentangled Conditional Diffusion Model for Multi-Contrast MRI Super-Resolution. 

The repository is modified based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). 

## SetUp Python Package Environment:
```
pip install -e .
```

## Download Dataset:
* [Raw IXI dataset](https://brain-development.org/ixi-dataset/)
* [Pre-processed IXI dataset](https://bit.ly/3yethO4): 500 for Training; 6 for Validation; 70 for Testing. T1-weighted images are co-registered to T2-weighted images using [FLIRT](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT).

## Download Pretrained Models:
 * [2x and 4x models trained on IXI dataset](https://drive.google.com/drive/folders/1h_bmH0ELEAIu8Z7hkUKerBom-SXyZ8i5?usp=sharing).

## Model Training
```
bash train_job.sh
```

## Model Testing
```
bash test_job.sh
```
