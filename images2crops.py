import numpy as np
import matplotlib.pyplot as plt


def preprocess_images(imgs_raw, subtract_mask=None, pos_neg_thresholds=None, oversaturation_thresh=2 ** 16 - 1):
    """
    The preprocessing of image stacks is composed of the following steps:
    1. Substract a mask [the mask is based on the median]
    2. Filter out images where the mean of their positive values is no more than some theresholds, and the same for their negative values
    3. Add a constant to make the images positive
    4. Return images and also the kept indices
    """
    N_images_orig = imgs_raw.shape[0]

    if (subtract_mask is None) or (pos_neg_thresholds is None):  # auto calculation from stack if no mask was given
        subtract_mask, pos_neg_thresholds = gen_params_for_preprocess(imgs_raw,percentile=85,
                                                                      oversaturation_thresh=oversaturation_thresh)

    # later - parallelize or move this to gpu
    imgs = imgs_raw.astype(np.float32)
    imgs_med_sub = imgs - subtract_mask
    mean_pos_arr = np.mean(np.maximum(imgs_med_sub, 0), axis=(1, 2))
    mean_neg_arr = np.mean(np.minimum(imgs_med_sub, 0), axis=(1, 2))
    imgs_med_sub_positive = imgs_med_sub + np.max(subtract_mask)

    inds_to_keep_mask = (mean_pos_arr <= pos_neg_thresholds[0]) * (mean_neg_arr >= pos_neg_thresholds[1])

    imgs_out = np.minimum(imgs_med_sub_positive[inds_to_keep_mask, ...], oversaturation_thresh).astype(np.uint16)
    inds_out = np.where(inds_to_keep_mask)[0]

    print(f"Kept {len(inds_out)}/{N_images_orig} images in the preprocessing ({len(inds_out) / N_images_orig:.2f})")

    return imgs_out, inds_out


def gen_params_for_preprocess(imgs_raw, percentile=85, oversaturation_thresh=2 ** 16 - 1):
    """
    Generate parameters subtract_mask, pos_neg_thresholds for function preprocess_images.
    """

    median_initial = median_image(imgs_raw)
    imgs_med_sub_initial = imgs_raw.astype(np.float32) - median_initial
    mean_pos_arr = np.mean(np.maximum(imgs_med_sub_initial, 0), axis=(1, 2))
    mean_neg_arr = np.mean(np.minimum(imgs_med_sub_initial, 0), axis=(1, 2))
    pos_neg_thresholds = np.asarray([np.percentile(mean_pos_arr, percentile),
                                     np.percentile(mean_neg_arr, 100 - percentile)])
    imgs_after_filter = imgs_raw[(mean_pos_arr <= pos_neg_thresholds[0]) * (mean_neg_arr >= pos_neg_thresholds[1]), ...]
    subtract_mask = median_image(imgs_after_filter)

    if np.max(np.abs(subtract_mask)) > oversaturation_thresh / 6:
        print("Warning! The calculated median mask might have some major outliers!")

    return subtract_mask, pos_neg_thresholds


def blobs_to_crops(imgs, imgs_rgb, blobs_df, crop_dims=np.array([[-18, 6], [-5, 5], 3]), include_large_context=False):
    N_blobs = len(blobs_df)
    y_arr = blobs_df.y.values
    x_arr = blobs_df.x.values
    frame_arr = blobs_df.frame.values

    success_mask = np.zeros(N_blobs, dtype=bool)
    crop_arr = np.zeros([N_blobs, int(np.diff(crop_dims[0])), int(np.diff(crop_dims[1]))])
    crop_rgb_arr = np.zeros([N_blobs, int(np.diff(crop_dims[0])), int(np.diff(crop_dims[1])), crop_dims[2]])

    if include_large_context:
        success_mask_LC = np.zeros(N_blobs, dtype=bool)
        crop_arr_LC = np.zeros([N_blobs, 2 * int(np.diff(crop_dims[0])), 2 * int(np.diff(crop_dims[1]))])
        success_mask_XLC = np.zeros(N_blobs, dtype=bool)
        crop_arr_XLC = np.zeros([N_blobs, 10 * int(np.diff(crop_dims[0])), 10 * int(np.diff(crop_dims[1]))])

    for i in range(N_blobs):
        crop_arr[i], crop_rgb_arr[i], success_mask[i] = crop_blob(imgs[frame_arr[i]], imgs_rgb[frame_arr[i]], y_arr[i], x_arr[i], crop_dims)
        if include_large_context:
            crop_arr_LC[i], success_mask_LC[i] = crop_blob(imgs_rgb[frame_arr[i]], y_arr[i], x_arr[i], 2 * crop_dims)
            crop_arr_XLC[i], success_mask_XLC[i] = crop_blob(imgs_rgb[frame_arr[i]], y_arr[i], x_arr[i],
                                                             10 * crop_dims)

    crop_out = crop_arr[success_mask, ...]
    crop_rgb_out = crop_rgb_arr[success_mask, ...]
    if include_large_context:
        return crop_out, np.where(success_mask)[0], crop_arr_LC[success_mask, ...], crop_arr_XLC[success_mask, ...]
    else:
        return crop_out, crop_rgb_out, np.where(success_mask)[0]


def median_image(I):
    """
    Receive an array or stack of images and return an image of each pixel's median along the stack
    """
    I = np.asarray(I)
    shape = I.shape
    if len(shape) <= 2:
        median_image = I
    else:
        N_images, Ny, Nx = shape
        # Do a loop over rows instead of fully vectorized because otherwise this causes a serious memory problem!!!
        median_image = np.zeros(shape[1:])
        for i in range(Ny):
            median_image[i, ...] = np.median(I[:, i, ...], axis=0)
    return median_image


def crop_blob(img, img_rgb, blob_y, blob_x, crop_dims):
    height = int(np.diff(crop_dims[0]))
    width = int(np.diff(crop_dims[1]))
    N_y, N_x = img.shape[:2]

    # TOP BLOB!!
    y = int(np.round(blob_y))
    x = int(np.round(blob_x))
    y_min = y + crop_dims[0][0]
    y_max = y + crop_dims[0][1]
    x_min = x + crop_dims[1][0]
    x_max = x + crop_dims[1][1]

    if ((x_min < 0) | (y_min < 0)) | ((x_max > N_x - 1) | (y_max > N_y - 1)):
        crop = np.zeros([int(np.diff(crop_dims[0])), int(np.diff(crop_dims[1]))])
        crop_rgb = np.zeros([int(np.diff(crop_dims[0])), int(np.diff(crop_dims[1])), crop_dims[2]])
        success = False
    else:
        crop = img[y_min:y_max, x_min:x_max]
        crop_rgb = img_rgb[y_min:y_max, x_min:x_max]
        success = True

    return crop, crop_rgb, success
