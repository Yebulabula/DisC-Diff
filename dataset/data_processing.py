"""
    Preprocess BRATS2018 Data
"""
import h5py
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import warnings
import cv2
import numpy as np
import numpy.ma as ma
import torchvision.transforms
from alive_progress import alive_bar
import time
from torchvision.transforms import transforms
import os
from multiprocessing import Process

warnings.simplefilter('ignore')


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return image


def getLR(hr_data, scaling_factor):
    imgfft = np.fft.fftn(hr_data)
    imgfft = np.fft.fftshift(imgfft)

    x, y, z = imgfft.shape
    diff_x = x // (scaling_factor * 2)
    diff_y = y // (scaling_factor * 2)
    diff_z = z // (scaling_factor * 2)

    x_centre = x // 2
    y_centre = y // 2
    z_centre = z // 2

    mask = np.zeros(imgfft.shape)
    # mask[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y, :] = 1
    # imgfft = imgfft * mask

    imgfft = imgfft[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y, :]

    imgifft = np.fft.ifftshift(imgfft)
    imgifft = np.fft.ifftn(imgifft)
    img_out = abs(imgifft)
    return img_out


def N4_Bias_Field_Correction(img_obj, process=True):
    """
        (Optional): Perform BraTS MRI processing on the original .nii image object.
    """
    if not process:
        return img_obj
    else:
        maskImage = sitk.OtsuThreshold(img_obj, 0, 1, 200)
        image = sitk.Cast(img_obj, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFilltingLevels = 4
        corrector.SetMaximumNumberOfIterations([4] * numberFilltingLevels)
        output = corrector.Execute(image, maskImage)
        return output


def normalize(data):
    data = np.clip(np.fabs(data), -np.inf, np.inf)
    data -= np.amin(data)
    data /= np.amax(data)
    return data


def process(path, modality, scaling_factor, transform):
    # image = np.array(load_nib(path + modality + '.nii'))
    image = np.array(load_nib(path))
    hr_img = transform(image)
    lr_img = getLR(hr_img, scaling_factor=scaling_factor)
    hr_img = normalize(hr_img)
    lr_img = normalize(lr_img)
    return hr_img, lr_img


def get_hr_lr(start, end, folder_names, path, data_shape, scaling_factor, transform):
    hr_arr, lr_arr = [], []

    with alive_bar(len(folder_names[start:end]), force_tty=True) as bar:
        for idx in range(start, end):
            sub_path = "{}{}".format('../../IXI_T2_dataset/', folder_names[idx])

            hr_img, lr_img = process(sub_path, modality, scaling_factor=scaling_factor, transform=transform)

            if hr_img.shape == data_shape:
                hr_arr.append(hr_img)
                lr_arr.append(lr_img)
            else:
                print('problem subpath', sub_path)
            bar()

    hr_arr, lr_arr = np.array(hr_arr), np.array(lr_arr)

    hr_arr = np.moveaxis(hr_arr, -1, 1)
    lr_arr = np.moveaxis(lr_arr, -1, 1)
    return hr_arr, lr_arr


def create_npy_dataset(path, modality, output_path, data_shape, scaling_factor=2, transform=None):
    print(scaling_factor)
    folder_names = os.listdir(path)
    print('.DS_Store' in folder_names)
    hr_arr, lr_arr = np.empty([len(folder_names), *data_shape]), np.empty(
        [len(folder_names), *data_shape])

    training_hr, training_lr = get_hr_lr(0, 500, folder_names, path, data_shape, scaling_factor, transform)
    # with open(output_path + f'/IXI_training_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
    #     np.save(f, training_hr)

    with open(output_path + f'/IXI_training_lr_{modality}_scale_by_{scaling_factor}_imgs_small.npy', 'wb') as f:
        np.save(f, training_lr)

    valid_hr, valid_lr = get_hr_lr(500, 506, folder_names, path, data_shape, scaling_factor, transform)
    # with open(output_path + f'/IXI_valid_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
    #     np.save(f, valid_hr)
    with open(output_path + f'/IXI_valid_lr_{modality}_scale_by_{scaling_factor}_imgs_small.npy', 'wb') as f:
        np.save(f, valid_lr)

    testing_hr, testing_lr = get_hr_lr(506, 576, folder_names, path, data_shape, scaling_factor, transform)
    # with open(output_path + f'/IXI_testing_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
    #     np.save(f, testing_hr)

    with open(output_path + f'/IXI_testing_lr_{modality}_scale_by_{scaling_factor}_imgs_small.npy', 'wb') as f:
        np.save(f, testing_lr)

    print("=======Completed=======")


def load_nib(file_path):
    try:
        proxy = nib.load(file_path)
        data = proxy.get_fdata()
        proxy.uncache()
        return data
    except:
        print("Invalid file path is given")


if __name__ == "__main__":
    # filenames = os.listdir('/home/cbtil/Downloads/glioma500')
    # count = 0
    #
    # data_shape = (224, 224, 30)
    # transforms = torchvision.transforms.Compose([CenterCrop(data_shape)])
    # lr_T2_list = []
    # hr_T2_list = []
    # hr_T1_list = []
    # for i in filenames:
    #     initial_sub_folders = os.listdir(f'/home/cbtil/Downloads/glioma500/{i}')
    #     sub_folders = '\t'.join(initial_sub_folders)
    #     if 'T1.nii.gz' in sub_folders and 'T2.nii.gz' in sub_folders:
    #         T1 = None
    #         T2 = None
    #         for j in initial_sub_folders:
    #             if 'T1.nii.gz' in j:
    #                 T1 = np.array(load_nib(f'/home/cbtil/Downloads/glioma500/{i}/{j}'))
    #                 if T1.shape[0] >= 100 and T1.shape[1] >= 100 and T1.shape[2] >= 100:
    #                     T1 = transforms(T1)
    #
    #             if 'T2.nii.gz' in j:
    #                 T2 = np.array(load_nib(f'/home/cbtil/Downloads/glioma500/{i}/{j}'))
    #                 if T2.shape[0] >= 100 and T2.shape[1] >= 100 and T2.shape[2] >= 100:
    #                     T2 = transforms(T2)
    #
    #         if T1 is not None and T2 is not None:
    #             if T1.shape == (224, 224, 30) and T2.shape == (224, 224, 30):
    #                 count += 1
    #                 lr_T2 = getLR(T2, scaling_factor=4)
    #                 hr_T2 = normalize(T2)
    #                 lr_T2 = normalize(lr_T2)
    #                 hr_T1 = normalize(T1)
    #
    #                 print(hr_T1.shape, hr_T2.shape, lr_T2.shape)
    #                 hr_T1_list.append(hr_T1)
    #                 hr_T2_list.append(hr_T2)
    #                 lr_T2_list.append(lr_T2)
                    # f, axs = plt.subplots(1, 3, figsize=(15, 15))
                    # axs[0].imshow(lr_T2[..., 20].T, cmap='gray')
                    # axs[1].imshow(hr_T2[..., 20].T, cmap='gray')
                    # axs[2].imshow(hr_T1[..., 20].T, cmap='gray')
                    # plt.show()

    # outpath = '/home/cbtil/Documents/SRDIFF/guided-diffusion/datasets'

    # with open(outpath + f'/glioma_hr_T1_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, np.array(hr_T1_list))
    #
    # with open(outpath + f'/glioma_hr_T2_scale_by_{4}.npy', 'wb') as f:
    #     np.save(f, np.array(hr_T2_list))

    # with open(outpath + f'/glioma_lr_T2_scale_by_{4}_small.npy', 'wb') as f:
    #     np.save(f, np.array(lr_T2_list))

    # print(np.array(hr_T1_list).shape, np.array(hr_T2_list).shape, np.array(lr_T2_list).shape)
    # print(count)
    # sub_path = "{}{}".format('../../IXI_T2_dataset/', folder_names[idx])
    # image = np.array(load_nib(path))
    # modality = 't2'
    # data_shape = (224, 224, 96)
    # transforms = torchvision.transforms.Compose([CenterCrop(data_shape)])
    #
    # create_npy_dataset(modality=modality,
    #                    path='../../IXI_T2_dataset/',
    #                    output_path='new_dataset',
    #                    data_shape=data_shape,
    #                    scaling_factor=2,
    #                    transform=transforms)

    # hr_T1 = np.load('/home/cbtil/Documents/SRDIFF/guided-diffusion/datasets/glioma_hr_T1_scale_by_4.npy', mmap_mode="r")
    # hr_T2 = np.load('/home/cbtil/Documents/SRDIFF/guided-diffusion/datasets/glioma_hr_T2_scale_by_4.npy', mmap_mode="r")
    lr_T2_2 = np.load('/home/cbtil/Documents/SRDIFF/guided-diffusion/datasets/glioma_lr_T2_scale_by_2_small.npy', mmap_mode="r")
    lr_T2_4 = np.load('/home/cbtil/Documents/SRDIFF/guided-diffusion/datasets/glioma_lr_T2_scale_by_4_small.npy', mmap_mode="r")

    outpath = '/home/cbtil/Documents/SRDIFF/guided-diffusion/datasets'
    # with open(outpath + f'/glioma_training_hr_T1_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, hr_T1[:221])
    #
    # with open(outpath + f'/glioma_valid_hr_T1_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, hr_T1[221:251])
    #
    # with open(outpath + f'/glioma_testing_hr_T1_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, hr_T1[251:])
    #
    #
    # with open(outpath + f'/glioma_training_hr_T2_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, hr_T2[:221])
    #
    # with open(outpath + f'/glioma_valid_hr_T2_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, hr_T2[221:251])
    #
    # with open(outpath + f'/glioma_testing_hr_T2_scale_by_{4}_imgs.npy', 'wb') as f:
    #     np.save(f, hr_T2[251:])
    #
    #
    with open(outpath + f'/glioma_training_lr_T2_scale_by_{4}_imgs_small.npy', 'wb') as f:
        np.save(f, lr_T2_4[:221])

    with open(outpath + f'/glioma_valid_lr_T2_scale_by_{4}_imgs_small.npy', 'wb') as f:
        np.save(f, lr_T2_4[221:251])

    with open(outpath + f'/glioma_testing_lr_T2_scale_by_{4}_imgs_small.npy', 'wb') as f:
        np.save(f, lr_T2_4[251:])
    #
    #
    with open(outpath + f'/glioma_training_lr_T2_scale_by_{2}_imgs_small.npy', 'wb') as f:
        np.save(f, lr_T2_2[:221])

    with open(outpath + f'/glioma_valid_lr_T2_scale_by_{2}_imgs_small.npy', 'wb') as f:
        np.save(f, lr_T2_2[221:251])

    with open(outpath + f'/glioma_testing_lr_T2_scale_by_{2}_imgs_small.npy', 'wb') as f:
        np.save(f, lr_T2_2[251:])
    #
    #
    # f, axs = plt.subplots(1, 4, figsize=(15, 15))
    # axs[0].imshow(hr_T1[0,..., 20].T, cmap='gray')
    # axs[1].imshow(hr_T2[0,..., 20].T, cmap='gray')
    # axs[2].imshow(lr_T2_2[0,..., 20].T, cmap='gray')
    # axs[3].imshow(lr_T2_4[0,..., 20].T, cmap='gray')
    # plt.show()


