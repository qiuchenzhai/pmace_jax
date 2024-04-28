from pmace_jax.utils import *
from tqdm import trange


class PMACE():
    """This class is a decorator that can be used to prepare a function before it is called.

    Args:
        func (function): The function to be decorated.
        *args: Positional arguments to be passed to the decorated function.
        **kwargs: Keyword arguments to be passed to the decorated function.
    """
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self):
        def wrapper(*args, **kwargs):
            print('Preparing function ...')
            return_val = self.func(*args, **kwargs)
            return return_val

        return wrapper(*self.args, **self.kwargs)


@jit
def determine_denoising_area(cmplx_img, predefined_mask=None):
    """Determine the area for applying a denoiser based on a binary mask.

    Args:
        cmplx_img (jax.numpy.ndarray): Complex-valued image.
        predefined_mask (jax.numpy.ndarray, optional): Binary mask representing the region of interest for denoising.

    Returns:
        jax.numpy.ndarray: Array of row and column indices defining the denoising area [row_start, row_end, col_start, col_end].
    """
    # Initialize the predefined mask if not provided
    if predefined_mask is None:
        predefined_mask = jnp.ones_like(cmplx_img.real)

    # Compute the sum of the mask along rows and columns to find non-zero areas
    row_sums = jnp.sum(predefined_mask, axis=1)
    col_sums = jnp.sum(predefined_mask, axis=0)

    # Find the first and last non-zero elements in the sums to determine the bounding box
    row_start = jnp.argmax(row_sums > 0)
    row_end = cmplx_img.shape[0] - jnp.argmax(row_sums[::-1] > 0)
    col_start = jnp.argmax(col_sums > 0)
    col_end = cmplx_img.shape[1] - jnp.argmax(col_sums[::-1] > 0)

    # Define the denoising area using row and column indices
    area_crds = jnp.array([row_start, row_end, col_start, col_end])

    return area_crds


@jit
def calculate_mode_weights(probe_modes):
    """Calculate the weights for each mode in the probe.

    Args:
        probe_modes (jax.numpy.ndarray): Probe modes.

    Returns:
        jax.numpy.ndarray: Mode weights.
    """
    weights = jnp.linalg.norm(probe_modes, axis=(1, 2), ord='fro') ** 2
    total_weight = jnp.sum(weights)

    return weights / total_weight


@jit
def get_data_fit_pt(cur_est, joint_est, y_meas):
    """Calculate the data-fitting point for the current patch.

    Args:
        cur_est (jax.numpy.ndarray): Current estimated patch.
        joint_est (jax.numpy.ndarray): Joint estimated patch.
        y_meas (jax.numpy.ndarray): Current measurements.

    Returns:
        jax.numpy.ndarray: Data-fitting point.
    """
    # TODO: pmap
    # inv_f = compute_ift(y_meas * tmp_f)
    tmp_f = compute_ft(cur_est * joint_est)
    inv_f = compute_ift(y_meas * jnp.exp(1j * jnp.angle(tmp_f)) * jnp.abs(tmp_f))
    
    return divide_cmplx_numbers(inv_f, joint_est)


@jit
def squared_abs_ft(patches, mode):
    """Compute squared absolute value of the Fourier transform.

    Args:
        patches (jax.numpy.ndarray): Array of patches for object data fitting.
        mode (jax.numpy.ndarray): A single mode for object data fitting.

    Returns:
        jax.numpy.ndarray: Squared absolute values of the Fourier transform for the given input pair.
    """
    return jnp.abs(compute_ft(patches * mode)) ** 2


@jit
def object_data_fit_op(patches, modes, y_meas, data_fit_param, mode_weights):
    """Perform data fitting operation for the object using JAX and parallelize across probe modes.

    Args:
        patches (jax.numpy.ndarray): Patches for object data fitting.
        modes (jax.numpy.ndarray): Modes for object data fitting.
        y_meas (jax.numpy.ndarray): Real measurements for object data fitting.
        data_fit_param (float): Fitting parameter controlling the interpolation between current estimate and data-fitting point.
        mode_weights (jax.numpy.ndarray): Weights for each mode in the data fitting process.

    Returns:
        jax.numpy.ndarray: Estimated object after data fitting operation.
    """
    # Automatically vectorize squared_abs_ft over modes
    vectorized_squared_abs = vmap(squared_abs_ft, in_axes=(None, 0), out_axes=0)

    # Compute squared absolute values for all modes and sum them
    sum_abs_squared =  jnp.sum(vectorized_squared_abs(patches, modes), axis=0)

    # Normalize measurements based on sum of squared absolute values
    normalized_y_meas = y_meas / (jnp.sqrt(sum_abs_squared) + 1e-12)

    # Compute data fitting points for each mode and patch
    mode_contributions = vmap(get_data_fit_pt, in_axes=(0, None, 0), out_axes=0)(patches, modes, normalized_y_meas)

    # Transpose mode_contributions to get shape (num_nodes, num_patches, probe.shape[0], probe.shape[1])
    mode_contributions = jnp.transpose(mode_contributions, (1, 0, 2, 3))

    # Reshape mode_weights for broadcasting: from (2,) to (2, 1, 1, 1) to match mode_contributions
    mode_weights_reshaped = mode_weights[:, None, None, None]

    # Take weighted average between current estimates and closest data-fitting point
    output = (1 - data_fit_param) * patches + data_fit_param * jnp.sum(mode_weights_reshaped * mode_contributions, axis=0)

    return output


def bm4d_denoising(input_img, denoising_area_crds, bm3d_psd=0.1):
    """Perform BM4D denoising on a specified area of an input image. 

    Args:
        input_img (jax.numpy.ndarray): The input image as a JAX array.
        denoising_area_crds (jax.numpy.ndarray): Coordinates defining the rectangular area to denoise.
        bm3d_psd (float, optional): The PSD estimate for BM3D denoising, controlling the strength of denoising. 

    Returns:
        jax.numpy.ndarray: The denoised image.
    """
    # Convert the JAX array to a NumPy array for BM3D denoising
    img_numpy_array = np.asarray(input_img.block_until_ready())

    # Extract the denoising area from the input image
    denoising_area = img_numpy_array[denoising_area_crds[0]: denoising_area_crds[1],
                                     denoising_area_crds[2]: denoising_area_crds[3]]

    # Apply complex BM3D denoising to the denoising area
    denoised_area = bm4d(denoising_area, bm3d_psd, profile=BM4DProfileBM3DComplex())[:, :, 0]

    # Replace the denoised area in the input image
    img_numpy_array[denoising_area_crds[0]: denoising_area_crds[1],
                    denoising_area_crds[2]: denoising_area_crds[3]] = denoised_area

    return jnp.array(img_numpy_array)


def pixel_weighted_avg_op(patches, modes, patch_crds, image_sz, probe_exp=1.5, mode_weights=None,
                          regularization=False, bm3d_psd=0.1, blk_idx=None):
    """Calculate the pixel-weighted average of projected patches and reallocates the result.

    Args:
        patches (jax.numpy.ndarray): The image patches to be processed.
        modes (jax.numpy.ndarray): The modes associated with the probe, used to weight the patches during averaging.
        patch_crds (jax.numpy.ndarray): The coordinates for each patch in the full image.
        image_sz (tuple): The size of the output image.
        probe_exp (float, optional): The exponent used to calculate the weight for each mode. 
        mode_weights (jax.numpy.ndarray, optional): The weights for each mode. 
        regularization (bool, optional): Flag to enable or disable BM3D denoising as a regularization step. 
        bm3d_psd (float, optional): The PSD estimate for BM3D denoising.
        blk_idx (list of tuples, optional): Block indices for BM3D denoising.

    Returns:
        jax.numpy.ndarray: The reconciled image after weighted averaging and optional denoising.
        jax.numpy.ndarray: The patches extracted from the reconciled image.
    """    
    # Initialization
    output_img = jnp.zeros(image_sz, dtype=np.complex64)

    # Calculate the weights for each mode in the probe.
    if mode_weights is None:
        mode_weights = calcualte_mode_weights(modes)

    # Process each mode
    for mode_idx, cur_mode in enumerate(modes):
        patch_weight = jnp.abs(cur_mode) ** probe_exp
        image_weight = patch2img([patch_weight] * len(patches), patch_crds, image_sz)
        tmp_img = patch2img(patches * patch_weight, patch_crds, image_sz, image_weight)
        output_img = output_img + mode_weights[mode_idx] * tmp_img

    # Apply BM3D denoising if regularization is enabled.
    if regularization:
        output_patch = bm4d_denoising(output_img, blk_idx, bm3d_psd=bm3d_psd)

    # Convert the resulting complex image to patches.
    output_patch = img2patch(output_img, patch_crds)

    return output_img, output_patch


def update_dict(input_dictionary, key, value):
    """Update the dictionary with a new key-value pair.

    Args:
        input_dictionary (dict): The dictionary to be updated.
        key: The key to be added or updated in the dictionary.
        value: The value associated with the key.

    Returns:
        dict: The updated dictionary.
    """
    input_dictionary[key] = value

    return input_dictionary


def add_probe_mode(modes, patches, y_meas, probe_dict, energy_ratio=0.05,
                   fres_prop=False, dx=None, wavelength=None, propagation_distance=None):
    """Add a new probe mode to the existing list of probe modes and update the probe dictionary.

    Args:
        modes (jax.numpy.ndarray): Array of existing probe modes.
        patches (jax.numpy.ndarray): The estimates of projected patches.
        y_meas (jax.numpy.ndarray): The intensity measurement.
        probe_dict (dict): Dictionary containing probe modes.
        energy_ratio (float, optional): Ratio of energy in the new probe mode compared to the existing ones. Default is 0.05.
        fres_prop (bool, optional): Flag for performing Fresnel propagation. Default is False.
        dx (float, optional): Sampling interval at source plane.
        wavelength (float, optional): Wavelength of the imaging radiation.
        propagation_distance (float, optional): Propagation distance.

    Returns:
        jax.numpy.ndarray: Updated array of probe modes with the newly added probe mode.
        dict: Updated dictionary of probe modes with the newly added probe mode.
    """
    # Calculate the sum of estimated intensities
    vectorized_squared_abs = vmap(squared_abs_ft, in_axes=(None, 0), out_axes=0)
    sum_abs_squared =  jnp.sum(vectorized_squared_abs(patches, modes), axis=0)

    # Calculate the square root of the total energy
    # sqrt_total_energy = jnp.sqrt(jnp.sum(jnp.linalg.norm(mode) ** 2 for mode in modes))
    total_energy = 0
    for mode in modes:
        total_energy = total_energy + jnp.linalg.norm(mode)
    sqrt_total_energy = jnp.sqrt(total_energy)

    # Calculate the residual intensity value and apply clip-to-zero strategy
    res_meas = jnp.sqrt(jnp.maximum(y_meas ** 2 - sum_abs_squared, 0))

    # Calculate the new probe mode
    tmp_probe_arr = jnp.divide(compute_ift(res_meas), patches)
    tmp_probe_mode = jnp.average(tmp_probe_arr, axis=0)

    # Apply Fresnel propagation if requested
    if fres_prop:
        k0 = 2 * jnp.pi / wavelength
        Nx, Ny = y_meas.shape[-2], y_meas.shape[-1]
        # if dx is None:
        #     dx = jnp.sqrt(2 * jnp.pi * propagation_distance / (k0 * Nx))
        new_probe_mode = fresnel_propagation(tmp_probe_mode, wavelength, propagation_distance, dx)
        new_probe_arr = [new_probe_mode] * len(y_meas)
    else:
        new_probe_mode = tmp_probe_mode
        new_probe_arr = tmp_probe_arr

    # Update probe_dict and probe_modes
    probe_dict = update_dict(probe_dict, len(modes), jnp.array(new_probe_arr))
    modes = jnp.concatenate((modes, jnp.expand_dims(new_probe_mode, axis=0)), axis=0)

    # Balance the energy distribution among probe modes
    for mode_idx, cur_mode in enumerate(modes):
        cur_probe_arr = jnp.array(probe_dict[mode_idx])
        # probe_modes.at[mode_idx].set(jnp.average(new_probe_arr, axis=0))
        modes.at[mode_idx].set(cur_mode / jnp.sqrt(1 + energy_ratio))
        # modes[mode_idx] = cur_mode / jnp.sqrt(1 + energy_ratio)
        probe_dict[mode_idx] = cur_probe_arr / jnp.sqrt(1 + energy_ratio)

    return modes, probe_dict


@jit
def orthogonalize_images(cmplx_imgs):
    """Orthogonalize complex-valued images using Singular Value Decomposition (SVD).

    Args:
        cmplx_imgs (list of jax.numpy.ndarrays): List of input complex-valued images.

    Returns:
        list of jax.numpy.ndarrays: List of orthogonalized complex-valued images.
    """
    # Initialize an empty list
    orthogonalized_imgs = []

    # Stack the flattened images into a matrix
    stacked_imgs = jnp.stack(cmplx_imgs, axis=-1)

    # Reshape the stacked matrix to have each image as a column
    reshaped_imgs = jnp.reshape(stacked_imgs, (-1, len(cmplx_imgs)))

    # Perform SVD
    U, s, Vh = jnp.linalg.svd(reshaped_imgs, full_matrices=False)

    # Reconstruct the orthogonalized images using singular vectors
    recon_imgs = jnp.dot(U, jnp.diag(s))

    # Reshape orthogonalized images to their original shapes
    for i in range(len(cmplx_imgs)):
        ortho_img = jnp.reshape(recon_imgs[:, i], cmplx_imgs[i].shape)
        orthogonalized_imgs.append(ortho_img)

    return orthogonalized_imgs


def find_center_offset(cmplx_img):
    """Find the offset between the center of the input complex image and the true center of the image.

    Args:
        cmplx_img (jax.numpy.ndarray): Complex-valued input image.

    Returns:
        list: A list containing the offset in the x and y directions, respectively.
    """
    # Find the center of the given image
    c_0, c_1 = (jnp.shape(cmplx_img)[0] - 1)/ 2.0, (jnp.shape(cmplx_img)[1] - 1) / 2.0

    # Calculate peak and mean value of the magnitude image
    mag_img = jnp.abs(cmplx_img)
    peak_mag, mean_mag = jnp.amax(mag_img), jnp.mean(mag_img)

    # Find a group of points above the mean value
    pts = jnp.argwhere(jnp.logical_and(mag_img >= mean_mag, mag_img <= peak_mag))

    # Find the unknown shifted center by averaging the group of points
    curr_center = jnp.mean(pts, axis=0)

    # Compute the offset between the unknown shifted center and the true center of the image
    offset = [c_0 - curr_center[0], c_1 - curr_center[1]]


    return offset


def shift_complex_array(arr, delta_y, delta_x) :
    """Shift a 2D complex array by non-integer displacement amounts using the Fourier shift theorem.

    Args:
        arr (jax.numpy.ndarray): The image to be aligned/corrected.
        delta_y (jax.numpy.ndarray): The displacement amount in the y-direction.
        delta_x (jax.numpy.ndarray): The displacement amount in the x-direction.

    Returns:
        jax.numpy.ndarray: The shifted 2D complex array.
    """
    # Determine the dimensions of the input array
    ny, nx = arr.shape
    
    # Determine the padding width based on the maximum displacement and add a safety margin
    pad_width = max(int(1.5 * abs(delta_x)), int(1.5 * abs(delta_y)), 20)
    
    # Pad the input array with zeros
    padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)
    
    # Compute the 2D Fourier transform of the padded array
    arr_fft = np.fft.fft2(padded_arr)
    
    # Determine the dimensions of the padded array
    ly, lx = padded_arr.shape
    
    # Compute the frequency grid for the Fourier transform
    x = np.fft.fftfreq(lx)
    y = np.fft.fftfreq(ly)
    xv, yv = np.meshgrid(x, y)
    
    # Compute the phase shift kernel based on the desired displacement
    shift_kernel = np.exp(-2j * np.pi * (xv * delta_x + yv * delta_y))
    
    # Apply the phase shift to the Fourier transformed array
    arr_fft_shifted = arr_fft * shift_kernel
    
    # Compute the inverse Fourier transform to obtain the shifted array in the spatial domain
    arr_shifted_padded = np.fft.ifft2(arr_fft_shifted)
    
    # Determine the starting indices for the region of interest in the padded array
    startx = pad_width
    starty = pad_width
    
    # Extract the region of interest from the padded array to obtain the final shifted array
    arr_shifted = arr_shifted_padded[starty :starty + ny, startx :startx + nx]
    
    return arr_shifted


def correct_img_center(input_img, ref_img=None):
    """
    Correct the center of an image by aligning it with a reference image.

    Args:
        input_img (jnp.ndarray): The image to be aligned/corrected.
        ref_img (jnp.ndarray): The reference image. If not provided, the input image is used as the reference.

    Returns:
        jnp.ndarray: The corrected image with the center aligned to the reference.
    """
    # Check the reference image
    if ref_img is None:
        ref_img = jnp.copy(input_img)

    # Compute center offset using the reference image
    offset = find_center_offset(ref_img)

    # Shift the image back to the correct location
    output = shift_complex_array(input_img, offset[0], offset[1])

    return output


def pmace_recon(y_meas, patch_crds, init_object,
                init_probe=None, ref_object=None, ref_probe=None,
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None,
                object_data_fit_param=0.5, probe_data_fit_param=0.5, rho=0.5, probe_exp=1.5,
                regularization=False, reg_param=0.1, probe_center_correction=False,
                iter_add_mode=[], energy_ratio=0.05, iter_orthogonalize_modes=[],
                fres_prop=False, dx=None, wavelength=None, propagation_dist=None):
    """Projected Multi-Agent Consensus Equilibrium (PMACE).
    
    Args:
        y_meas (jax.numpy.ndarray): The measured intensity data.
        patch_crds (jax.numpy.ndarray)): Coordinates of the patches used for localized reconstructions.
        init_object (jax.numpy.ndarray): Initial guess of the object to be reconstructed.
        init_probe (jax.numpy.ndarray, optional): Initial guess of the probe, required for joint reconstruction.
        ref_object (jax.numpy.ndarray, optional): Reference object for calculating error metrics.
        ref_probe (jax.numpy.ndarray, optional): Reference probe for calculating error metrics.
        num_iter (int, optional): Number of iterations for the reconstruction process. Defaults to 100.
        joint_recon (bool, optional): Flag to enable joint reconstruction of the object and the probe. Defaults to False.
        recon_win (jax.numpy.ndarray, optional): Window function to specify the region of interest for reconstruction.
        save_dir (str, optional): Directory path to save intermediate and final reconstruction results.
        object_data_fit_param (float, optional): Data fitting parameter for the object reconstruction. Defaults to 0.5.
        probe_data_fit_param (float, optional): Data fitting parameter for the probe reconstruction. Defaults to 0.5.
        rho (float, optional): Mann averaging parameter for the iteration. Defaults to 0.5.
        probe_exp (float, optional): Exponent for probe mode weighting. Defaults to 1.5.
        regularization (bool, optional): Flag to enable regularization in the reconstruction. Defaults to False.
        reg_param (float, optional): Regularization parameter, applicable if regularization is True. Defaults to 0.1.
        iter_add_mode (list, optional): Iterations at which new modes are added for probe reconstruction.
        energy_ratio (float, optional): Energy ratio for adding new probe modes. Defaults to 0.05.
        iter_orthogonalize_modes (list, optional): Iterations at which probe modes are orthogonalized.
        fres_prop (bool, optional): Flag to enable Fresnel propagation in the reconstruction. Defaults to False.
        dx (float, optional): Sampling interval in the object plane, required if Fresnel propagation is enabled.
        wavelength (float, optional): Illumination wavelength, required if Fresnel propagation is enabled.
        propagation_dist (float, optional): Fresnel propagation distance from the object to the detector plane.

        
    Returns:
        dict: Reconstructed complex images and NRMSE between reconstructions and reference images.
            Keys:
                - 'object' (jax.numpy.ndarray): Reconstructed complex object.
                - 'probe' (jax.numpy.ndarray): Reconstructed complex probe.
                - 'err_obj' (list): NRMSE values for the object reconstructions.
                - 'err_probe' (list): NRMSE values for the probe reconstructions (if joint_recon is True).
                - 'err_meas' (list): NRMSE values for the measured data.
    """
    approach = 'reg-PMACE' if regularization else 'PMACE'
    cdtype = jnp.complex64

    # Create save directory
    if save_dir is not None:
        # Ensure the save directory ends with a '/'
        save_dir = save_dir if save_dir.endswith('/') else save_dir + '/'
        os.makedirs(save_dir, exist_ok=True)

    #
    if init_object is None:
        raise ValueError('Initial object not specified.')
    if init_probe is None and ref_probe is None:
        raise ValueError('Initial probe and reference probe not specified.')
    if recon_win is None:
        recon_win = jnp.ones_like(init_object)

    # Determine the reconstruction or denoising region
    denoising_blk_idx = determine_denoising_area(init_object, predefined_mask=recon_win)

    # Initialize error metrics
    nrmse_object = []
    nrmse_probe = [ [] for _ in range(max(2, 1 + len(set(iter_add_mode)))) ]
    nrmse_meas = []

    # Initialize estimates with specific data type and creat current patches
    est_object = jnp.array(init_object, dtype=cdtype)
    new_patch = img2patch(est_object, patch_crds)
    image_sz = est_object.shape

    # Expand the dimensions of reference probe
    if ref_probe is not None:
        if jnp.ndim(jnp.array(ref_probe)) == 2:
            ref_probe_modes = jnp.expand_dims(jnp.array(ref_probe, dtype=cdtype), axis=0)  # [num_mode, mode_h, mode_w]
        elif jnp.ndim(jnp.array(ref_probe)) > 2:
            ref_probe_modes = jnp.array(ref_probe, dtype=cdtype)
    else:
        ref_probe_modes = None

    # Initialize probe
    if joint_recon:
        if jnp.ndim(jnp.array(init_probe)) == 2:
            probe_modes = jnp.expand_dims(init_probe, axis=0)
        elif jnp.ndim(jnp.array(init_probe)) == 3:
            probe_modes = jnp.array(init_probe, dtype=cdtype)
        # Initialize probe dictionary
        probe_dict = {}
        for mode_idx, cur_mode in enumerate(probe_modes):
            new_probe_arr = jnp.array([cur_mode] * len(y_meas))
            probe_dict[mode_idx] = new_probe_arr
    else:
        probe_modes = ref_probe_modes

    # Iterate over the number of iterations
    print('{} recon starts ...'.format(approach))
    for i in trange(num_iter):
        # Calculate the weight for each probe mode
        mode_weights = calculate_mode_weights(probe_modes)

        # Update the current patch using data-fitting operator
        cur_patch = object_data_fit_op(new_patch, probe_modes, y_meas,
                                       object_data_fit_param, mode_weights)

        est_obj, consens_patch = pixel_weighted_avg_op(2 * cur_patch - new_patch, probe_modes, patch_crds, image_sz, 
                                                       probe_exp=probe_exp, mode_weights=mode_weights,
                                                       regularization=regularization, bm3d_psd=reg_param, blk_idx=denoising_blk_idx)

        new_patch = new_patch + 2 * rho * (consens_patch - cur_patch)

        if not regularization:
            est_obj, consens_patch = pixel_weighted_avg_op(new_patch, probe_modes, patch_crds, image_sz, 
                                                           probe_exp=probe_exp, mode_weights=mode_weights)

        if joint_recon:
            # Add new probe mode
            if i + 1 in iter_add_mode:
                probe_modes, probe_dict = add_probe_mode(probe_modes, consens_patch, y_meas, probe_dict, 
                                                         energy_ratio=energy_ratio, fres_prop=fres_prop, 
                                                         dx=dx, wavelength=wavelength, propagation_distance=propagation_dist)
                save_tiff(probe_modes[-1], save_dir + 'added_mode_iter_{}.tiff'.format(i+1))

            # Orthogonalize probe modes
            if i + 1 in iter_orthogonalize_modes:
                probe_modes = jnp.array(orthogonalize_images(probe_modes))
                # Update probe array
                for mode_idx, cur_mode in enumerate(probe_modes):
                    probe_dict[mode_idx] = jnp.array([cur_mode] * len(y_meas))

            # Automatically vectorize squared_abs_ft over modes
            vectorized_squared_abs = vmap(squared_abs_ft, in_axes=(None, 0), out_axes=0)

            # Compute squared absolute values for all modes and sum them
            sum_abs_squared =  jnp.sum(vectorized_squared_abs(consens_patch, probe_modes), axis=0)

            # Normalize measurements based on sum of squared absolute values
            normalized_y_meas = y_meas / (jnp.sqrt(sum_abs_squared) + 1e-12)

            # Compute data fitting points for each mode and patch, this returns (num_patches, num_nodes, probe.shape[0], probe.shape[1])
            data_fitting_modes = vmap(get_data_fit_pt, in_axes=(None, 0, 0), out_axes=0)(probe_modes, consens_patch, normalized_y_meas)

            # Transpose mode_contributions to get shape (num_nodes, num_patches, probe.shape[0], probe.shape[1])
            data_fitting_modes = jnp.transpose(data_fitting_modes, (1, 0, 2, 3))

            # Loop through probe_modes to update each mode
            for mode_idx, cur_mode in enumerate(probe_modes):
                # Get the current probe data
                # new_probe_arr = probe_dict[mode_idx]
                new_probe_arr = jnp.array([cur_mode] * len(y_meas))

                # Apply the probe data fitting operation: w <- F(v; w)
                cur_probe_arr = (1 - probe_data_fit_param) * new_probe_arr + probe_data_fit_param * data_fitting_modes[mode_idx]

                # Calculate the consensus probe: z <- G(2w - v)
                consens_probe = np.average((2 * cur_probe_arr - new_probe_arr), axis=0)
                # Probe center correction
                # TODO: Subpixel shift after integer shift
                if probe_center_correction:
                    consens_probe = correct_img_center(consens_probe)

                # Update the probe data: v <- v + 2 * rho * (z - w)
                new_probe_arr = new_probe_arr + 2 * rho * (consens_probe - cur_probe_arr)

                # Update probe modes
                probe_modes = probe_modes.at[mode_idx].set(jnp.average(new_probe_arr, axis=0))
                probe_dict[mode_idx] = new_probe_arr
                
        # Phase normalization and scale image to minimize the intensity difference
        if ref_object is not None:
            revy_obj = phase_norm(est_obj * recon_win, ref_object * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_object * recon_win, cstr=recon_win)
            nrmse_object.append(err_obj)
        else:
            revy_obj = est_obj

        # Phase normalization and scale probe to minimize the intensity difference
        if joint_recon and (ref_probe_modes is not None):
            revy_probe = []
            for mode_idx in range(min(len(probe_modes), len(ref_probe_modes))):
                tmp_probe_mode = phase_norm(jnp.copy(probe_modes[mode_idx]), ref_probe_modes[mode_idx])
                tmp_probe_err = compute_nrmse(tmp_probe_mode, ref_probe_modes[mode_idx])
                nrmse_probe[mode_idx].append(tmp_probe_err)
                # revy_probe.append(tmp_probe_mode)
                # revy_probe = jnp.array(revy_probe)
        # else:
        #     revy_probe = probe_modes
        revy_probe = probe_modes
        
        # Calculate error in measurement domain
        vectorized_squared_abs = vmap(squared_abs_ft, in_axes=(None, 0), out_axes=0)
        sum_abs_squared = jnp.sum(vectorized_squared_abs(consens_patch, probe_modes), axis=0)
        nrmse_meas.append(compute_nrmse(jnp.sqrt(sum_abs_squared), y_meas))

        if save_dir is not None and (i+1) % 20 == 0:
            # SAVE INTER RESULT
            save_tiff(est_obj, f'{save_dir}est_obj_iter_{i + 1}.tiff')
            for mode_idx, cur_mode in enumerate(probe_modes):
                save_tiff(cur_mode, f'{save_dir}est_probe_mode_{mode_idx}_iter_{i + 1}.tiff')

    # Save recon results
    if save_dir is not None:
        # Save estimated object if provided
        if est_obj is not None:
            save_tiff(est_obj, f'{save_dir}est_obj_iter_{i + 1}.tiff')
        # Save NRMSE for object and measurements if provided
        if nrmse_object is not None:
            save_array(nrmse_object, f'{save_dir}nrmse_obj_{nrmse_object[-1]}')
        if nrmse_meas is not None:
            save_array(nrmse_meas, f'{save_dir}nrmse_meas_{nrmse_meas[-1]}')

    # Save joint reconstruction results if applicable
    if joint_recon and probe_modes is not None:
        for mode_idx, cur_mode in enumerate(probe_modes):
            save_tiff(cur_mode, f'{save_dir}probe_est_mode_{mode_idx}_iter_{i + 1}.tiff')
        # Save NRMSE for probe if provided
        # print(nrmse_probe)
        if nrmse_probe is not None:
            if len(nrmse_probe) == len(probe_modes):
                for mode_idx, nrmse_mode in enumerate(nrmse_probe):
                    if nrmse_mode:
                        save_array(nrmse_mode, f'{save_dir}probe_mode_{mode_idx}_nrmse_{nrmse_mode[-1]}')
            else:
                save_array(nrmse_probe, f'{save_dir}nrmse_probe_{nrmse_probe[-1]}')
            
    # Return recon results
    print('{} recon completed.'.format(approach))
    keys = ['object', 'probe', 'err_obj', 'err_probe', 'err_meas']
    vals = [revy_obj, revy_probe, nrmse_object, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output