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


def pmace_recon(y_meas, patch_crds, init_object,
                init_probe=None, ref_object=None, ref_probe=None,
                num_iter=100, joint_recon=False, recon_win=None, save_dir=None,
                object_data_fit_param=0.5, probe_data_fit_param=0.5, rho=0.5, probe_exp=1.5,
                regularization=False, reg_param=0.1,
                iter_add_mode=[], energy_ratio=0.05, iter_orthogonalize_modes=[],
                fresnel_propagation=False, dx=None, wavelength=None, propagation_dist=None):
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
        fresnel_propagation (bool, optional): Flag to enable Fresnel propagation in the reconstruction. Defaults to False.
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
    nrmse_probe = []
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

    # # Initialize probe
    # if joint_recon:
    #     if jnp.ndim(jnp.array(init_probe)) == 2:
    #         probe_modes = jnp.expand_dims(init_probe, axis=0)
    #     elif jnp.ndim(jnp.array(init_probe)) == 3:
    #         probe_modes = jnp.array(init_probe, dtype=cdtype)
    #     probe_dict = {}
    #     for mode_idx, cur_mode in enumerate(probe_modes):
    #         new_probe_arr = jnp.array([cur_mode] * len(y_meas))
    #         probe_dict = jax.ops.index_update(probe_dict, jax.ops.index[mode_idx], new_probe_arr)  # {mode_idx: mode_array}
    # else:
    #     probe_modes = ref_probe_modes
    # TODO: Initialize probe modes
    revy_probe = ref_probe_modes
    probe_modes = ref_probe_modes

    # Iterate over the number of iterations
    print('{} recon starts ...'.format(approach))
    for i in trange(num_iter):
        # Calculate the weight for each probe mode
        mode_weights = calculate_mode_weights(probe_modes)

        # Update the current patch using data-fitting operator
        cur_patch = object_data_fit_op(new_patch, probe_modes, y_meas,
                                       object_data_fit_param, mode_weights)

        est_obj, consens_patch = pixel_weighted_avg_op(2 * cur_patch - new_patch, probe_modes,
                                                       patch_crds, image_sz, probe_exp=probe_exp, mode_weights=mode_weights,
                                                       regularization=regularization, bm3d_psd=reg_param, blk_idx=denoising_blk_idx)

        new_patch = new_patch + 2 * rho * (consens_patch - cur_patch)

        if not regularization:
            est_obj, consens_patch = pixel_weighted_avg_op(new_patch, probe_modes, patch_crds, image_sz, probe_exp=probe_exp, mode_weights=mode_weights)

        # Phase normalization and scale image to minimize the intensity difference
        if ref_object is not None:
            revy_obj = phase_norm(est_obj * recon_win, ref_object * recon_win, cstr=recon_win)
            err_obj = compute_nrmse(revy_obj * recon_win, ref_object * recon_win, cstr=recon_win)
            nrmse_object.append(err_obj)
        else:
            revy_obj = est_obj

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
    if joint_recon and probe_modes is not None and save_dir is not None:
        for mode_idx, cur_mode in enumerate(probe_modes):
            save_tiff(cur_mode, f'{save_dir}probe_est_mode_{mode_idx}_iter_{i + 1}.tiff')
       # Save NRMSE for probe if provided
        if nrmse_probe is not None:
            if len(nrmse_probe) == len(probe_modes):
                save_array(nrmse_probe, f'{save_dir}nrmse_probe{nrmse_probe[-1]}')
            else:
                for mode_idx, nrmse_mode in enumerate(nrmse_probe):
                    if nrmse_mode:
                        save_array(nrmse_mode, f'{save_dir}probe_mode_{mode_idx}_nrmse_{nrmse_mode[-1]}')

    # Return recon results
    print('{} recon completed.'.format(approach))
    keys = ['object', 'probe', 'err_obj', 'err_probe', 'err_meas']
    vals = [revy_obj, revy_probe, nrmse_object, nrmse_probe, nrmse_meas]
    output = dict(zip(keys, vals))

    return output