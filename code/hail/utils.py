
from torch.nn import functional as F
import numpy as np

def reparameterize_logit(logit):
    import warnings
    warnings.filterwarnings('ignore', message='.*Mixed memory format inputs detected.*')
    beta = F.gumbel_softmax(logit, tau=1.0, dim=1, hard=True)
    return beta

def divide_into_batches(in_tensor, num_batches):
    batch_size = in_tensor.shape[0] // num_batches
    remainder = in_tensor.shape[0] % num_batches
    batches = []

    current_start = 0
    for i in range(num_batches):
        current_end = current_start + batch_size
        if remainder:
            current_end += 1
            remainder -= 1
        batches.append(in_tensor[current_start:current_end, ...])
        current_start = current_end
    return batches


def normalize_intensity(image):
    thresh = np.percentile(image.flatten(), 95)
    image = image / (thresh + 1e-5)
    image = np.clip(image, a_min=0.0, a_max=5.0)
    return image, thresh


def zero_pad(image, image_dim=256):
    [n_row, n_col, n_slc] = image.shape
    image_padded = np.zeros((image_dim, image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2] = image
    return image_padded

def zero_pad2d(image, image_dim=256):
    [n_row, n_col] = image.shape
    image_padded = np.zeros((image_dim, image_dim))
    center_loc = image_dim // 2
    image_padded[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2] = image
    return image_padded


def crop(image, n_row, n_col, n_slc):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2,
                 center_loc - n_slc // 2: center_loc + n_slc - n_slc // 2]

def crop2d(image, n_row, n_col):
    image_dim = image.shape[0]
    center_loc = image_dim // 2
    return image[center_loc - n_row // 2: center_loc + n_row - n_row // 2,
                 center_loc - n_col // 2: center_loc + n_col - n_col // 2]