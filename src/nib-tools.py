"""Helper functions for processing 3D MRI brain scans."""
from os import fdopen, remove

import nibabel as nib
import numpy as np
from shutil import move
from tempfile import mkstemp



def normalize(target_array, max_val=255):
    """Normalize array values from 0 to 255."""
    target_array -= target_array.min()
    normalized_array = max_val * np.divide(target_array, target_array.max())

    return normalized_array


def read_im(image_path):
    """Read nii from path, return nii volume and its data."""
    nii_img = nib.load(image_path)
    nii_data = nii_img.get_data()

    return nii_data, nii_img


def replace(file_path, pattern, subst):
    """
    Replace strings in a file.
    
    Parametrs:
        pattern (str, iterable): pattern to replace.
        subst (str, iterable): subtitution.
    
    """
    #Create temp file
    fh, abs_path = mkstemp()
    with fdopen(fh,'w') as new_file:
        with open(file_path) as old_file:
            for line in old_file:
                new_file.write(line.replace(pattern, subst))
    #Remove original file
    remove(file_path)
    #Move new file
    move(abs_path, file_path)
