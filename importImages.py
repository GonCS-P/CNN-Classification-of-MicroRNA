# Module importImages
# This module reads and saves images. Temporarily the preprocess file-based function is also here.
import numpy as np
from PIL import Image
from os.path import join
import tifffile
from csbdeep.io import save_tiff_imagej_compatible
from utils import include_file_extension, backup_before_save, save_dict
from config import paths
from images2crops import preprocess_images
from denoising import denoise_images


def read_tif(imgpath):
    """
    Read a (possibly stacked) tif image and return as an ndarray of ndarrays.
    Requires the FULL path to the image
    """
    imagesArray = np.asarray(tifffile.imread(imgpath))  # read with tifffile
    return imagesArray


def read_tif_from_folder(filename, folder='raw'):
    """
    Read a tif from the selected folder according to config/paths. recommended to use without .tif extension
    """
    return read_tif(join(paths[folder], include_file_extension(filename)))


def save_tif(imgpath, img, axes="TYX"):
    """
    Save tif at the full path. axes argument is for tif stack
    """
    backup_before_save(imgpath)
    save_tiff_imagej_compatible(imgpath, img, axes=axes)


def save_tif_at_folder(filename, img, folder='raw', axes="TYX"):
    """
    Save images as tif at a specified folder (see config.py file for valid folder names)
    """
    imgpath = join(paths[folder], include_file_extension(filename))
    save_tif(imgpath, img, axes=axes)
    return imgpath


def preprocess_image_file(filename, subtract_mask=None, pos_neg_thresholds=None, add_to_images=0, multiply_images_by=1):
    """Preprocess a raw image file. Can either load a mask and thresholds or calculate them (if input is None)"""
    imgs = read_tif_from_folder(filename, "raw")

    imgs = (multiply_images_by * imgs + add_to_images).astype(np.uint16)
    if subtract_mask is not None:
        subtract_mask = read_tif_from_folder(subtract_mask,
                                             "preprocessing")

    if pos_neg_thresholds is not None:
        pos_neg_thresholds = np.load(
            join(paths["preprocessing"], include_file_extension(pos_neg_thresholds, "npy")))

    imgs_preprocessed, kept_indices = preprocess_images(np.reshape(imgs, imgs.shape[:3]),
                                                        subtract_mask=subtract_mask,
                                                        pos_neg_thresholds=pos_neg_thresholds
                                                        )
    save_tif_at_folder(filename, imgs_preprocessed, "preprocessed")

    new2orig_indices_dict = {i: kept_indices[i] for i in range(len(kept_indices))}
    dict_path = join(paths['logs'], include_file_extension(filename)[:-4] + "_indices_dict.npy")
    backup_before_save(dict_path)
    save_dict(new2orig_indices_dict, dict_path)

    print(f"Saved preprocessed image {filename} in the preprocessed images directory")


def denoise_image_file(images_file_name, model_name):
    """Denoise a preprocessed image file and save the output in the denoised image directory."""
    imgs = read_tif_from_folder(images_file_name, 'preprocessed')
    denoised_imgs = denoise_images(imgs, model_name)
    save_tif_at_folder(images_file_name, denoised_imgs, 'denoised')
    print(f"Saved denoised image {images_file_name} in the denoised images directory")
