import sys
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torchvision.transforms import ToTensor

from skimage.filters import threshold_otsu
from skimage.morphology import isotropic_closing

from model import HAIL
from utils import crop, zero_pad, zero_pad2d, crop2d


def background_removal(image_vol):
    [n_row, n_col, n_slc] = image_vol.shape
    thresh = threshold_otsu(image_vol)
    mask = (image_vol >= thresh) * 1.0
    mask = zero_pad(mask, 256)
    mask = isotropic_closing(mask, radius=20)
    mask = crop(mask, n_row, n_col, n_slc)
    image_vol[mask < 1e-4] = 0.0
    return image_vol

def background_removal2d(image_vol):
    [n_row, n_col] = image_vol.shape
    thresh = threshold_otsu(image_vol)
    mask = (image_vol >= thresh) * 1.0
    mask = zero_pad2d(mask, 256)
    mask = isotropic_closing(mask, radius=20)
    mask = crop2d(mask, n_row, n_col)
    image_vol[mask < 1e-4] = 0.0
    return image_vol

def obtain_single_image(image_path, bg_removal=True):
    image_obj = nib.Nifti1Image.from_filename(image_path)
    image_vol = np.array(image_obj.get_fdata().astype(np.float32))
    thresh = np.percentile(image_vol.flatten(), 95)
    image_vol = image_vol / (thresh + 1e-5)
    image_vol = np.clip(image_vol, a_min=0.0, a_max=5.0)
    if bg_removal:
        image_vol = background_removal(image_vol)

    n_row, n_col, n_slc = image_vol.shape
    # zero padding
    image_padded = np.zeros((224, 224, 224)).astype(np.float32)
    image_padded[112 - n_row // 2:112 + n_row // 2 + n_row % 2,
                 112 - n_col // 2:112 + n_col // 2 + n_col % 2,
                 112 - n_slc // 2:112 + n_slc // 2 + n_slc % 2] = image_vol
    return ToTensor()(image_padded), image_obj.header, thresh

def load_source_images(image_paths, bg_removal=True):
    source_images = []
    image_header = None
    for image_path in image_paths:
        image_vol, image_header, _ = obtain_single_image(image_path, bg_removal)
        source_images.append(image_vol.float().permute(2, 1, 0))
    return source_images, image_header


if __name__ == '__main__':
    #args = sys.argv[1:] if args is None else args
    parser = argparse.ArgumentParser(description='Harmonization Across Imaging Location(HAIL)')
    parser.add_argument('--in-path', type=Path, action='append', required=True)
    parser.add_argument('--target-image', type=Path, action='append', default=[])
    parser.add_argument('--out-path', type=Path, action='append', required=True)
    parser.add_argument('--harmonization-model', type=Path, default=Path('/tmp/model_weights/harmonization.pt'))
    parser.add_argument('--fusion-model', type=Path, default=Path('/tmp/model_weights/fusion.pt'))
    parser.add_argument('--beta-dim', type=int, default=5)
    parser.add_argument('--theta-dim', type=int, default=2)
    parser.add_argument('--save-intermediate', action='store_true', default=False)
    parser.add_argument('--intermediate-out-dir', type=Path, default=Path.cwd())
    parser.add_argument('--no-bg-removal', dest='bg_removal', action='store_false', default=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--num-batches', type=int, default=4)

    args = parser.parse_args()
    #args = parser.parse_args(args)
    print(args)

    text_div = '=' * 10
    print(f'{text_div} BEGIN HARMONIZATION {text_div}')

    # ==== GET ABSOLUTE PATHS ====
    for argname in ['in_path', 'target_image', 'out_path', 'harmonization_model',
                    'fusion_model', 'intermediate_out_dir']:
        if isinstance(getattr(args, argname), list):
            setattr(args, argname, [path.resolve() for path in getattr(args, argname)])
        else:
            setattr(args, argname, getattr(args, argname).resolve())

    if args.save_intermediate:
        mkdir_p(args.intermediate_out_dir)

    # ==== INITIALIZE MODEL ====
    hail = HAIL(beta_dim=args.beta_dim,
                  theta_dim=args.theta_dim,
                  eta_dim=2,
                  pretrained=args.harmonization_model,
                  gpu_id=args.gpu_id)

    # ==== LOAD SOURCE IMAGES ====
    source_images, image_header = load_source_images(args.in_path, args.bg_removal)

    # ==== LOAD TARGET IMAGES IF PROVIDED ====
   
    target_images, norm_vals = [], []
    for target_image_path, out_path in zip(args.target_image, args.out_path):
        target_image_tmp, tmp_header, norm_val = obtain_single_image(target_image_path, args.bg_removal)
        target_images.append(target_image_tmp.permute(2, 1, 0).permute(0, 2, 1).flip(1)[100:120, ...])
        norm_vals.append(norm_val)
        if args.save_intermediate:
            out_prefix = out_path.name.replace('.nii.gz', '')
            save_img = target_image_tmp.permute(1, 2, 0).numpy()[112 - 96:112 + 96, :, 112 - 96:112 + 96]
            target_obj = nib.Nifti1Image(save_img * norm_val, None, tmp_header)
            target_obj.to_filename(args.intermediate_out_dir / f'{out_prefix}_target.nii.gz')
    if args.save_intermediate:
        out_prefix = args.out_path[0].name.replace('.nii.gz', '')
        with open(args.intermediate_out_dir / f'{out_prefix}_targetnorms.txt', 'w') as fp:
            fp.write('image,norm_val\n')
            for i, norm_val in enumerate(norm_vals):
                fp.write(f'{i},{norm_val:.6f}\n')
        np.savetxt(args.intermediate_out_dir / f'{out_prefix}_targetnorms.txt', norm_vals)
    target_theta = None
    target_eta = None
    
    # ===== BEGIN HARMONIZATION=====
    hail.harmonize(
        source_images=[image.permute(2, 0, 1) for image in source_images],
        target_images=target_images,
        target_theta=target_theta,
        target_eta=target_eta,
        out_paths=args.out_path,
        header=image_header,
        recon_orientation='axial',
        norm_vals=norm_vals,
        num_batches=args.num_batches,
        save_intermediate=args.save_intermediate,
        intermediate_out_dir=args.intermediate_out_dir,
    )