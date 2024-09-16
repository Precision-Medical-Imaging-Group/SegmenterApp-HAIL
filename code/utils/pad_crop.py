import ants
import nibabel as nib
import numpy as np


def pad_or_crop(image, target_shape=(192, 224, 192)):
    data = image.get_fdata()
    current_shape = data.shape
    
    # Calculate the padding or cropping needed
    pad_width = [(max(0, target - current) // 2, max(0, target - current) - max(0, target - current) // 2)
                 for target, current in zip(target_shape, current_shape)]
    
    # Pad the data if necessary
    data_padded = np.pad(data, pad_width, mode='constant', constant_values=data.min())
    
    # Crop if the padding went beyond the target shape (in cases of negative padding)
    crop_slices = tuple(slice((pad[0] - (data_padded.shape[i] - target_shape[i])) if data_padded.shape[i] > target_shape[i] else pad[0],
                              pad[0] + target_shape[i]) for i, pad in enumerate(pad_width))
    
    data_cropped = data_padded[crop_slices]
    data_cropped = data_padded[:-1, 2:226, :-1]
    
    return nib.Nifti1Image(data_cropped, image.affine)

moving_image = ants.image_read('/media/abhijeet/DataThunder1/BraTS2024_Data/brats2024_ped_val/harm_test_inp/practical_ellis.nii.gz')
fixed_image = ants.image_read('/home/abhijeet/Code/mni_icbm152_nl_VI_nifti/icbm_avg_152_t1_tal_nlin_symmetric_VI.nii')

moving_image = ants.resample_image(moving_image, (1,1,1))
fixed_image = ants.resample_image(fixed_image, (1,1,1))
result = ants.registration(fixed_image, moving_image, type_of_transform = 'SyN', verbose = True)


ants.image_write(result['warpedmovout'], '/media/abhijeet/DataThunder1/BraTS2024_Data/brats2024_ped_val/harm_test_inp/practical_ellis.nii.gz')
nifti_file_path = '/media/abhijeet/DataThunder1/BraTS2024_Data/brats2024_ped_val/harm_test_inp/practical_ellis.nii.gz'
image = nib.load(nifti_file_path)
image = pad_or_crop(image)
nib.save(image, '/media/abhijeet/DataThunder1/BraTS2024_Data/brats2024_ped_val/harm_test_inp/reg_practical_ellis.nii.gz')
