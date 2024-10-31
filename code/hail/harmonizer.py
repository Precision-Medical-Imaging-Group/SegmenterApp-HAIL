import sys
import argparse
from pathlib import Path
from os import PathLike

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import ToTensor

from skimage.filters import threshold_otsu
from skimage.morphology import isotropic_closing

from model import HAIL
from utils import crop, zero_pad, zero_pad2d, crop2d
from typing import List, Tuple

"""
The harmonizer module is used to harmonize the images across different imaging locations 
"""

def background_removal(image_vol: np.ndarray)-> np.ndarray:
    """remove background from the head using otsu threhold for MRI

    Args:
        image_vol (np.ndarray): brain 3D image volume

    Returns:
        np.ndarray: background removed 3D image volume
    """
    [n_row, n_col, n_slc] = image_vol.shape
    thresh = threshold_otsu(image_vol)
    mask = (image_vol >= thresh) * 1.0
    mask = zero_pad(mask, 256)
    mask = isotropic_closing(mask, radius=20)
    mask = crop(mask, n_row, n_col, n_slc)
    image_vol[mask < 1e-4] = 0.0
    return image_vol

def background_removal2d(image_vol: np.ndarray) ->  np.ndarray:
    """remove background from the head using otsu threhold for MRI

    Args:
        image_vol (np.ndarray): brain 2D slice

    Returns:
        np.ndarray: background removed 2D slice
    """
    [n_row, n_col] = image_vol.shape
    thresh = threshold_otsu(image_vol)
    mask = (image_vol >= thresh) * 1.0
    mask = zero_pad2d(mask, 256)
    mask = isotropic_closing(mask, radius=20)
    mask = crop2d(mask, n_row, n_col)
    image_vol[mask < 1e-4] = 0.0
    return image_vol

def obtain_single_image(image_path: PathLike, bg_removal: bool=True, a_max:float=5.0) -> Tuple[torch.Tensor, nib.Nifti1Header, Tuple[float, float]]:
    """get a single image from a NifTi file, and preprocess it for harmonization

    Args:
        image_path (Pathlike): path to the NifTi file
        bg_removal (bool, optional): remove background from the image. Defaults to True.
        a_max (float, optional): maximum value for the image. Defaults to 5.0.

    Returns:
        Tuple[torch.Tensor, nib.Nifti1Header, Tuple[float, float]]: preprocessed image, image header, and normalization values
    """
    image_obj = nib.Nifti1Image.from_filename(image_path)
    image_vol = np.array(image_obj.get_fdata().astype(np.float32))
    thresh = np.percentile(image_vol.flatten(), 95)
    max_thresh = image_vol.max()
    image_vol = image_vol / (thresh + 1e-5)
    image_vol = np.clip(image_vol, a_min=0.0, a_max=a_max)
    if bg_removal:
        image_vol = background_removal(image_vol)

    n_row, n_col, n_slc = image_vol.shape
    # zero padding
    image_padded = np.zeros((224, 224, 224)).astype(np.float32)
    image_padded[112 - n_row // 2:112 + n_row // 2 + n_row % 2,
                 112 - n_col // 2:112 + n_col // 2 + n_col % 2,
                 112 - n_slc // 2:112 + n_slc // 2 + n_slc % 2] = image_vol
    return ToTensor()(image_padded), image_obj.header, (thresh, max_thresh)

def load_source_images(image_paths:List[PathLike], bg_removal:bool =True) -> Tuple[List[torch.Tensor], nib.Nifti1Header]:
    """ Load all the source images from the list of paths

    Args:
        image_paths (List[Pathlike]): list of paths to the images
        bg_removal (bool, optional): remove background from the image. Defaults to True.

    Returns:
        Tuple[List[torch.Tensor], nib.Nifti1Header]: list of preprocessed images and the image header
    """
    
    source_images = []
    image_header = None
    for image_path in image_paths:
        image_vol, image_header, _ = obtain_single_image(image_path, bg_removal)
        source_images.append(image_vol.float().permute(2, 1, 0))
    return source_images, image_header


if __name__ == '__main__':
    """
    python harmonizer.py --in-path /path/to/image1.nii.gz /path/to/image2.nii.gz --target-image /path/to/target_image.nii.gz --out-path /path/to/output.nii.gz
    
    """
    parser = argparse.ArgumentParser(description='Harmonization Across Imaging Location(HAIL)')
    parser.add_argument('--in-path', type=Path, action='append', required=True)
    parser.add_argument('--target-image', type=Path, action='append', default=[])
    parser.add_argument('--out-path', type=Path, action='append', required=True)
    parser.add_argument('--harmonization-model', type=Path, default=Path('/tmp/model_weights/harmonization.pt'))
    parser.add_argument('--fusion-model', type=Path, default=Path('/tmp/model_weights/fusion.pt'))
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--no-bg-removal', dest='bg_removal', action='store_false', default=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=1)

    args = parser.parse_args()
    print(args)

    text_div = '=' * 10
    print(f'{text_div} BEGIN HARMONIZATION {text_div}')

    # get all the absolute paths
    for argname in ['in_path', 'target_image', 'out_path', 'harmonization_model',
                    'fusion_model']:
        if isinstance(getattr(args, argname), list):
            setattr(args, argname, [path.resolve() for path in getattr(args, argname)])
        else:
            setattr(args, argname, getattr(args, argname).resolve())


    print(args)
    # initialize the HAIL model
    hail = HAIL(beta_dim=args.beta_dim,
                  theta_dim=args.theta_dim,
                  eta_dim=2,
                  pretrained=args.harmonization_model,
                  gpu_id=args.gpu_id)

    # load source images
    source_images, image_header = load_source_images(args.in_path, args.bg_removal)

    # load target template teams
    target_images, norm_vals = [], []
    for target_image_path, out_path in zip(args.target_image, args.out_path):
        target_image_tmp, tmp_header, norm_val = obtain_single_image(target_image_path, args.bg_removal, a_max=6.0)
        target_images.append(target_image_tmp.permute(2, 1, 0).permute(0, 2, 1).flip(1)[100:120, ...])
        norm_vals.append(norm_val)


    
    # begin the harmonization process to generate the harmonized image in a axial fashion
    hail.harmonize(
        source_images=[image.permute(2, 0, 1) for image in source_images],
        target_images=target_images,
        target_theta=None,
        target_eta=None,
        out_paths=args.out_path,
        header=image_header,
        recon_orientation='axial',
        norm_vals=norm_vals,
        num_batches=args.num_batches,
    )
    print(f'{text_div} END HARMONIZATION {text_div}')