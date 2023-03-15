from setuptools import setup

setup(
    name="guided-diffusion",
    py_modules=["guided_diffusion"],
    install_requires=["torch", "tqdm", "matplotlib","numpy","torchvision","scikit-image", "scipy", "blobfile", "cv2"],
)
