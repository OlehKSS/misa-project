"""Helper functions for processing 3D MRI brain scans."""
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


# numeric labels for brain tissue types
CSF_label = 1
GM_label = 2
WM_label = 3

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


def show_slice(img, slice_no):
    """
        Inputs: img (nibabel): image name
                slice_no (np slice): np.s_[:, :, 30]
    """
    data = img.get_fdata()
    plt.figure()
    plt.imshow(data[slice_no].T, cmap='gray')
    plt.show()
    

def show_slice_data(data, slice_no):
    """
    Display slice of a given array.
    """
    plt.imshow(data[slice_no], cmap = "gray")
    plt.show()


def calc_dice(segmented_images, groundtruth_images):
    """
    Calcualte dice similarity coefficient between two volumes.
    """
    segData = segmented_images + groundtruth_images
    TP_value = np.amax(segmented_images) + np.amax(groundtruth_images)
    
    # found a true positive: segmentation result and groundtruth match(both are positive)
    TP = (segData == TP_value).sum()
    segData_FP = 2. * segmented_images + groundtruth_images
    segData_FN = segmented_images + 2. * groundtruth_images
    
    # found a false positive: segmentation result and groundtruth mismatch
    FP = (segData_FP == 2 * np.amax(segmented_images)).sum() 
    
    # found a false negative: segmentation result and groundtruth mismatch
    FN = (segData_FN == 2 * np.amax(groundtruth_images)).sum() 
    
    return 2*TP/(2*TP+FP+FN)  # according to the definition of DICE similarity score


def dice_similarity(segmented_img, groundtruth_img):
    """
    Extract binary label images for regions
        Inputs: segmented_img (nibabel): segmented labels nii file
                groundtruth_img (nibabel): groundtruth labels nii file
        Returns: DICE_index (float): Dice similarity score between the two images (nii files)        
    """
    
    segmented_data = segmented_img.get_data().copy()
    groundtruth_data = groundtruth_img.get_data().copy()
    seg_CSF = (segmented_data == CSF_label) * 1
    gt_CSF = (groundtruth_data == CSF_label) * 1
    seg_GM = (segmented_data == GM_label) * 1
    gt_GM = (groundtruth_data == GM_label) * 1
    seg_WM = (segmented_data == WM_label) * 1
    gt_WM = (groundtruth_data == WM_label) * 1
    
    dice_CSF = calc_dice(seg_CSF, gt_CSF)
    dice_GM = calc_dice(seg_GM, gt_GM)
    dice_WM = calc_dice(seg_WM, gt_WM)
    
    
    return dice_CSF, dice_GM, dice_WM


def apply_mask(target_data, gt_data):
    """
    Create mask using groundtruth image and apply it.
    
    Inputs: 
        gt_img: groundtuth mask
        target_img: raw data, apply mask to it
    
    Returns: 
        masked_img: target image with mask applied (background removed)
    """
    
    
    # Create mask: Select pixels higher than 0 in gt and set to 1
    gt_data[gt_data > 0] = 1
    
    # Apply mask
    target_data = np.multiply(target_data, gt_data)
    
    
    return target_data


def seg_data_to_nii(original_im, y_pred, features_nonzero_row_indicies):
    """
        Inputs: original_im (nibabel): original image nii file
                y_pred (np array): labels for all non-zero points
                features_nonzero_row_indicies (np array): indicies of non-zero points,
                                                          same length as y_pred
        Returns: segment_nii (nibabel): segmented labels nii file        
    """
    original_img_shape = original_im.get_data().shape
    original_img_len = original_img_shape[0] * original_img_shape[1] * original_img_shape[2]
    segment_im = np.zeros(original_img_len)
    labels = np.copy(y_pred) + 1
    segment_im[features_nonzero_row_indicies] = labels
    segment_im = np.reshape(segment_im, original_im.shape)
    segment_nii = nib.Nifti1Image(segment_im, original_im.affine, original_im.header)
    
    return segment_nii


def integrate_atlas_nii(original_im, y_pred, features_nonzero,
                        features_nonzero_row_indicies, weights, csf_atlas, gm_atlas,
                        wm_atlas):
    """
    Transforms segmenation result to nii file, puts correct labels in place.
    The segmentation labels should be: 1) CSF (darkest) 2) GM (middle) 3) WM (light)
    
    Inputs: 
    original_im (nibabel): original image nii file
    y_pred (np array): labels for all non-zero points
    features_nonzero (np array): feature vector of only non-zero intensities
    features_nonzero_row_indicies (np array): indicies of non-zero points,
                                              same length as y_pred

    Returns:
    segment_nii (nibabel): segmented labels nii file        
    """
    
    # Create image with all 3 classes and random labels
    y_pred = y_pred + 1
    original_img_shape = original_im.get_data().shape
    original_img_len = original_img_shape[0] * original_img_shape[1] * original_img_shape[2]
    
    segment_im = np.zeros(original_img_len)
    segment_im[features_nonzero_row_indicies] = y_pred
    segment_im = np.reshape(segment_im, original_im.shape)
    
    temp_class1_im = np.zeros_like(segment_im)
    temp_class2_im = np.zeros_like(segment_im)
    temp_class3_im = np.zeros_like(segment_im)
    
    #Assign class1 to 1
    temp_class1_im[segment_im == 1] = 1
    #Assign class2 to 2
    temp_class2_im[segment_im == 2] = 1
    #Assign class3 to 1
    temp_class3_im[segment_im == 3] = 1
    
    # Compute DICE between each class to determine which class it belongs to
    dice1 = [calc_dice(temp_class1_im, csf_atlas), calc_dice(temp_class2_im, csf_atlas), 
                                  calc_dice(temp_class3_im, csf_atlas)]
    dice2 = [calc_dice(temp_class1_im, wm_atlas), calc_dice(temp_class2_im, wm_atlas), 
                                  calc_dice(temp_class3_im, wm_atlas)]
    dice3 = [calc_dice(temp_class1_im, gm_atlas), calc_dice(temp_class2_im, gm_atlas), 
                                  calc_dice(temp_class3_im, gm_atlas)]
    csf_to_change = np.argmax(dice1) + 1
    wm_to_change = np.argmax(dice2) + 1
    gm_to_change = np.argmax(dice3) + 1
    
    
    #New y_pred
    y_pred_corrected_labels = np.zeros_like(y_pred)
    #Assign CSF to its correct label
    y_pred_corrected_labels[y_pred == csf_to_change] = CSF_label
    #Assign GM to its correct label
    y_pred_corrected_labels[y_pred == gm_to_change] = GM_label
    #Assign WM to its correct label
    y_pred_corrected_labels[y_pred == wm_to_change] = WM_label
    
    # Get weights back into original shape
    weight_csf_im = np.zeros(original_img_len)
    weight_gm_im = np.zeros(original_img_len)
    weight_wm_im = np.zeros(original_img_len)
    
    weight_csf_im[features_nonzero_row_indicies] = weights[:,csf_to_change-1]
    weight_gm_im[features_nonzero_row_indicies] = weights[:,gm_to_change-1]
    weight_wm_im[features_nonzero_row_indicies] = weights[:,wm_to_change-1]
    weight_csf_im = np.reshape(weight_csf_im, original_im.shape)
    weight_gm_im = np.reshape(weight_gm_im, original_im.shape)
    weight_wm_im = np.reshape(weight_wm_im, original_im.shape)
    
    # Multiply weights by each atlas
    csf_probs = weight_csf_im * csf_atlas
    gm_probs = weight_gm_im * gm_atlas
    wm_probs = weight_wm_im * wm_atlas
    
    # Assign GM, WM, CSF to voxel with highest probability
    GM = GM_label * np.nan_to_num((gm_probs > csf_probs) * (gm_probs > wm_probs))
    WM = WM_label * np.nan_to_num((wm_probs > csf_probs) * (wm_probs > gm_probs))
    CSF = CSF_label * np.nan_to_num((csf_probs > wm_probs) * (csf_probs > gm_probs))
    seg_im = GM + WM + CSF
    
    segment_im = np.zeros(original_img_len)
    segment_im = np.reshape(seg_im, original_im.shape)
    segment_nii = nib.Nifti1Image(segment_im, original_im.affine, original_im.header)

    return segment_nii


def seg_correct_labels_to_nii(original_im, y_pred, features_nonzero,
                              features_nonzero_row_indicies, csf_atlas, gm_atlas,
                              wm_atlas):
    """
    Transforms segmenation result to nii file, puts correct labels in place.
    The segmentation labels should be: 1) CSF (darkest) 2) GM (middle) 3) WM (light)
    
    Inputs: 
    original_im (nibabel): original image nii file
    y_pred (np array): labels for all non-zero points
    features_nonzero (np array): feature vector of only non-zero intensities
    features_nonzero_row_indicies (np array): indicies of non-zero points,
                                              same length as y_pred

    Returns:
    segment_nii (nibabel): segmented labels nii file        
    """
    
    # Create image with all 3 classes and random labels
    y_pred = y_pred + 1
    original_im_flat = original_im.get_data().copy().flatten()
    segment_im = np.zeros_like(original_im_flat)
    segment_im[features_nonzero_row_indicies] = y_pred
    segment_im = np.reshape(segment_im, original_im.shape)
    
    temp_class1_im = np.zeros_like(segment_im)
    temp_class2_im = np.zeros_like(segment_im)
    temp_class3_im = np.zeros_like(segment_im)
    
    #Assign class1 to 1
    temp_class1_im[segment_im == 1] = 1
    #Assign class2 to 2
    temp_class2_im[segment_im == 2] = 1
    #Assign class3 to 1
    temp_class3_im[segment_im == 3] = 1
    
    # Compute DICE between each class to determine which class it belongs to
    dice1 = [calc_dice(temp_class1_im, csf_atlas), calc_dice(temp_class2_im, csf_atlas), 
                                  calc_dice(temp_class3_im, csf_atlas)]
    dice2 = [calc_dice(temp_class1_im, wm_atlas), calc_dice(temp_class2_im, wm_atlas), 
                                  calc_dice(temp_class3_im, wm_atlas)]
    dice3 = [calc_dice(temp_class1_im, gm_atlas), calc_dice(temp_class2_im, gm_atlas), 
                                  calc_dice(temp_class3_im, gm_atlas)]
    csf_to_change = np.argmax(dice1) + 1
    wm_to_change = np.argmax(dice2) + 1
    gm_to_change = np.argmax(dice3) + 1
    
    
    #New y_pred
    y_pred_corrected_labels = np.zeros_like(y_pred)
    #Assign CSF to its correct label
    y_pred_corrected_labels[y_pred == csf_to_change] = CSF_label
    #Assign GM to its correct label
    y_pred_corrected_labels[y_pred == gm_to_change] = GM_label
    #Assign WM to its correct label
    y_pred_corrected_labels[y_pred == wm_to_change] = WM_label

    original_im_flat = original_im.get_data().copy().flatten()
    segment_im = np.zeros_like(original_im_flat)
    labels = np.copy(y_pred_corrected_labels)
    segment_im[features_nonzero_row_indicies] = labels
    segment_im = np.reshape(segment_im, original_im.shape)
    segment_nii = nib.Nifti1Image(segment_im, original_im.affine, original_im.header)

    return segment_nii
