
from typing import List, Tuple

import torch
from torch.nn import functional as F
import numpy as np

def reparameterize_logit(logit: torch.Tensor) -> torch.Tensor:
    """ reparameterize the logit tensor

    Args:
        logit (torch.Tensor): logit tensor

    Returns:
        torch.Tensor: reparameterized logit
    """
    import warnings
    warnings.filterwarnings('ignore', message='.*Mixed memory format inputs detected.*')
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta

def divide_into_batches(in_tensor: torch.Tensor, num_batches:int ) -> List[torch.Tensor]:
    """ divide the input tensor into multiple batches

    Args:
        in_tensor (torch.Tensor): tensor to be divided into batches
        num_batches (int): the number of batches to divide the tensor into

    Returns:
        List[torch.Tensor]: list of tensors divided into batches
    """
    batch_size = in_tensor.shape[0] // num_batches
    remainder = in_tensor.shape[0] % num_batches
    batches = []

    current_start = 0
    # divide the tensor into batches
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(in_tensor[current_start:current_end, ...])
        current_start = current_end
    return batches


def normalize_intensity(image: np.ndarray) -> Tuple[np.ndarray, int]:
    """ Normalize the intensity of the image

    Args:
        image (np.ndarray): input image

    Returns:
        Tuple[np.ndarray, int]: normalized image and the threshold value
    """

    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh


def zero_pad(image: np.ndarray, image_dim: int=256) -> np.ndarray:
    """ pad a 3D image with zeros with given image dimension

    Args:
        image (np.ndarray): input 3D image
        image_dim (int, optional): image dim. Defaults to 256.

    Returns:
        np.ndarray: padded image
    """
    [n_row, n_col, n_slc] = image.shape
    image_padded = np.zeros((image_dim, image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2] = image
    return image_padded

def zero_pad2d(image: np.ndarray, image_dim: int=256) -> np.ndarray:
    """ pad a 2D image with zeros with given image dimension

    Args:
        image (np.ndarray): input 2D image
        image_dim (int, optional): image dim. Defaults to 256.

    Returns:
        np.ndarray: padded image
    """
    [n_row, n_col] = image.shape
    image_padded = np.zeros((image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2] = image
    return image_padded


def crop(image: np.ndarray, n_row: int, n_col: int, n_slc: int) -> np.ndarray:
    """ crop a 3D image to the given dimensions

    Args:
        image (np.ndarray): input 3D image
        n_row (int): number of rows
        n_col (int): number of columns
        n_slc (int): number of slices

    Returns:
        np.ndarray: cropped image
    """
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2]

def crop2d(image: np.ndarray, n_row: int, n_col: int) -> np.ndarray:
    """ crop a 2D image to the given dimensions

    Args:
        image (np.ndarray): input 2D image
        n_row (int): number of rows
        n_col (int): number of columns

    Returns:
        np.ndarray: cropped image
    """
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2]