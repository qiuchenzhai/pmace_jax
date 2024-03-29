import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import tifffile as tiff
import imagecodecs

import pandas as pd
import re

from scipy.ndimage import gaussian_filter
from jax.numpy.fft import fft2, ifft2, fftfreq, fftshift


@jit
def compute_ft(input_array):
    """Compute the 2D Discrete Fourier Transform (DFT) of an input array.

    Args:
        input_array (jax.numpy.ndarray): The input 2D array for DFT computation.

    Returns:
        jax.numpy.ndarray: The result of the 2D DFT.
    """
    # DFT using jax.numpy
    a = jnp.fft.fftshift(input_array.astype(jnp.complex64), axes=(-2, -1))
    b = jnp.fft.fft2(a, s=None, axes=(-2, -1), norm='ortho')
    output = jnp.fft.ifftshift(b, axes=(-2, -1))

    return output.astype(jnp.complex64)


@jit
def compute_ift(input_array):
    """Compute the 2D Inverse Discrete Fourier Transform (IDFT) of an input array.

    Args:
        input_array (jax.numpy.ndarray): The input 2D array for IDFT computation.

    Returns:
        jax.numpy.ndarray: The result of the 2D IDFT.
    """
    # IDFT using jax.numpy
    a = jnp.fft.fftshift(input_array.astype(jnp.complex64), axes=(-2, -1))
    b = jnp.fft.ifft2(a, s=None, axes=(-2, -1), norm='ortho')
    output = jnp.fft.ifftshift(b, axes=(-2, -1))

    return output.astype(jnp.complex64)


@jit
def divide_cmplx_numbers(cmplx_num, cmplx_denom):
    """Perform element-wise division with complex numbers, handling division by zero.

    Args:
        cmplx_num (jax.numpy.ndarray): Complex numerator.
        cmplx_denom (jax.numpy.ndarray): Complex denominator.

    Returns:
        jax.numpy.ndarray: Result of the division.
    """
    # Use epsilon to avoid division by zero
    fro_norm = jnp.sqrt(jnp.sum(jnp.square(jnp.abs(cmplx_denom))))

    # Small constant for numerical stability
    epsilon = 1e-6 * fro_norm / jnp.sqrt(cmplx_denom.size)

    # Calculate the inverse of the denominator, considering epsilon
    denom_inv = jnp.conj(cmplx_denom) / (cmplx_denom * jnp.conj(cmplx_denom) + epsilon)

    # Perform element-wise operation, handling division by zero
    output = cmplx_num * denom_inv

    return output


def patch2img(img_patches, patch_coords, img_sz, weight=None):
    """Project image patches to a full-sized image with weights.

    Args:
        img_patches (jax.numpy.ndarray): Projected image patches.
        patch_coords (jax.numpy.ndarray): Coordinates of projections.
        img_sz (tuple): Size of the full image.
        weight (jax.numpy.ndarray): Normalization weight.

    Returns:
        jax.numpy.ndarray: The full-sized complex image.
    """
    # Initialization
    if weight is None:
        weight = jnp.ones(img_sz, dtype=jnp.complex64)
    full_img = jnp.zeros(img_sz, dtype=jnp.complex64)

    # Back projection
    for j in range(len(img_patches)):
        row_start, row_end, col_start, col_end = patch_coords[j]
        full_img = full_img.at[row_start:row_end, col_start:col_end].add(img_patches[j])

    # Normalization
    output = divide_cmplx_numbers(full_img, weight)

    return output


def single_patch_extraction(full_img, coords):
    """Extract a single patch from a full-size image based on given coordinates.

    Args:
        full_img (jax.numpy.ndarray): The full-size image from which to extract the patch.
        coords (jax.numpy.ndarray): Coordinates specifying the patch to extract.

    Returns:
        jax.numpy.ndarray: The extracted image patch.
    """
    row_start, row_end, col_start, col_end = coords
    start_indices = (row_start, col_start)
    slice_size = (row_end - row_start, col_end - col_start)
    
    return lax.dynamic_slice(jnp.array(full_img), start_indices, slice_size)


def img2patch(full_img, patch_coords):
    """Extract multiple patches from a full-size image.

    Args:
        full_img (jax.numpy.ndarray): The full-size image from which patches are to be extracted.
        patch_coords (jax.numpy.ndarray): An array of coordinates for each patch, specifying the region to extract
                                          for each patch in the full-size image.

    Returns:
        jax.numpy.ndarray: An array of extracted image patches.
    """
    patches = [single_patch_extraction(full_img, coords) for coords in patch_coords]
    
    return jnp.stack(patches)



def load_img(img_dir):
    """Load a complex image from a TIFF file.

    Args:
        img_dir (str): Directory of the TIFF image.

    Returns:
        numpy.ndarray: Complex image array.
    """
    # Read the TIFF image; assumes the TIFF file has multiple pages
    img = tiff.imread(img_dir)

    # Check if the image has at least 2 slices for real and imaginary parts
    if img.shape[0] < 2:
        raise ValueError("TIFF image does not contain enough slices for real and imaginary parts.")

    # Separate real and imaginary parts, magnitude, and phase
    # mag, pha = img[2], img[3]
    real, imag = img[0], img[1]

    # Combine real and imaginary parts to form the complex image
    cmplx_img = real + 1j * imag

    return jnp.array(cmplx_img)


def gen_scan_loc(cmplx_obj, cmplx_probe, num_pt, probe_spacing, randomization=True, max_offset=5, rng_key=random.PRNGKey(0)):
    """Generate scan locations.

    Args:
        cmplx_obj (jax.numpy.ndarray): Complex sample image to be scanned.
        cmplx_probe (jax.numpy.ndarray): Complex probe.
        num_pt (int): Number of scan points.
        probe_spacing (float): Probe spacing between neighboring scan positions.
        randomization (bool): Option to add random offsets to each scan point.
        max_offset (int): Maximum offsets to be added to scan points along each dimension.
        rng_key (jax.random.PRNGKey): Random number generator key for JAX.

    Returns:
        jax.numpy.ndarray: Generated scan points.
    """
    # Get image dimensions
    x, y = cmplx_obj.shape
    m, n = cmplx_probe.shape

    # Calculate number of points along each dimension
    num_pt_x, num_pt_y = int(jnp.sqrt(num_pt)), int(jnp.sqrt(num_pt))

    # Generate scan points in raster order
    scan_pt = jnp.array([((i - num_pt_x / 2 + 1 / 2) * probe_spacing + x / 2,
                          (j - num_pt_y / 2 + 1 / 2) * probe_spacing + y / 2)
                         for j in range(num_pt_x)
                         for i in range(num_pt_y)])

    # Add random offsets to each scan point
    if randomization:
        if rng_key is None:
            raise ValueError("rng_key must be provided for randomization.")
        offset = random.uniform(rng_key, shape=(num_pt, 2), minval=-max_offset, maxval=max_offset + 1)
        scan_pt += offset

    # Check if scanning points are beyond the valid region
    if jnp.any((jnp.amin(scan_pt) - m / 2) < 0) or jnp.any((jnp.amax(scan_pt) + n / 2) >= jnp.max(jnp.array([x, y]))):
        print('Warning: Scanning beyond the valid region! Please extend the image or reduce probe spacing.')

    return scan_pt


def gen_syn_data(cmplx_obj, cmplx_probe, patch_bounds,
                 add_noise=True, peak_photon_rate=1e5, shot_noise_pm=0.5, rng_key=random.PRNGKey(0), save_dir=None):
    """Simulate ptychographic intensity measurements.

    Args:
        cmplx_obj (jax.numpy.ndarray): Complex object.
        cmplx_probe (jax.numpy.ndarray): Complex probe.
        patch_bounds (list of tuple): Scan coordinates of projections.
        add_noise (bool): Option to add noise to data.
        peak_photon_rate (float): Peak rate of photon detection at the detector.
        shot_noise_pm (float): Expected number of Poisson distributed dark current noise.
        rng_key (jax.random.PRNGKey): Random key for JAX's random number generation.
        save_dir (str): Directory for saving generated data.

    Returns:
        jax.numpy.ndarray: Simulated ptychographic data.
    """
    # Expand the dimensions of complex-valued probe
    probe_modes = jnp.expand_dims(cmplx_probe, axis=0) if cmplx_probe.ndim == 2 else cmplx_probe

    # Get image dimensions
    m, n = probe_modes[0].shape
    num_pts = len(patch_bounds)

    # Extract patches from full-sized object
    projected_patches = img2patch(cmplx_obj, patch_bounds)

    # Initialize data array
    noiseless_data = jnp.zeros_like(projected_patches, dtype=jnp.float32)

    # Take 2D DFT and generate noiseless measurements
    for probe_mode in probe_modes:
        noiseless_data += jnp.abs(compute_ft(probe_mode * projected_patches)) ** 2

    # Introduce photon noise
    if add_noise:
        # Get peak signal value
        peak_signal_val = noiseless_data.max()

        # Calculate expected photon rate given peak signal value and peak photon rate
        expected_photon_rate = noiseless_data * peak_photon_rate / peak_signal_val

        # Poisson random values realization
        photon_count = random.poisson(rng_key, expected_photon_rate, shape=(num_pts, m, n))

        # Add dark current noise
        shot_noise = random.poisson(rng_key, lam=shot_noise_pm, shape=(num_pts, m, n))
        noisy_data = photon_count + shot_noise
        
        # Return the noisy data
        output = jnp.asarray(noisy_data, dtype=int)
    else:
        # Return the floating-point numbers without Poisson random variable realization
        output = jnp.asarray(noiseless_data)

    # Check directories and save simulated data
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for j in range(num_pts):
            tiff.imwrite(os.path.join(save_dir, f'frame_data_{j}.tiff'), np.array(output[j]))

    return output


def get_proj_coords_from_data(scan_loc, y_meas):
    """Calculate projection coordinates from scan points.

    Args:
        scan_loc (jax.numpy.ndarray): Scan locations.
        y_meas (jax.numpy.ndarray): Pre-processed measurements.

    Returns:
        jax.numpy.ndarray: Scan coordinates.
    """
    num_pts, m, n = y_meas.shape
    rounded_scan_loc = jnp.round(scan_loc)

    projection_coords = jnp.zeros((num_pts, 4), dtype=int)
    projection_coords = projection_coords.at[:, 0].set(rounded_scan_loc[:, 1] - m // 2)
    projection_coords = projection_coords.at[:, 1].set(rounded_scan_loc[:, 1] + m // 2)
    projection_coords = projection_coords.at[:, 2].set(rounded_scan_loc[:, 0] - n // 2)
    projection_coords = projection_coords.at[:, 3].set(rounded_scan_loc[:, 0] + n // 2)
    
    return projection_coords


def load_measurement(fpath):
    """Read measurements from path and pre-process data.

    Args:
        fpath: file directory.

    Returns:
        pre-processed measurement (square root of non-negative data).
    """
    # Specify the order of measurement
    def key_func(fname):
        non_digits = re.compile("\D")
        output = int(non_digits.sub("", fname))
        return output

    # Read the measurements and remove negative values
    work_dir = os.listdir(fpath)
    if '.DS_Store' in work_dir:
        work_dir.remove('.DS_Store')
    measurements = jnp.array([tiff.imread(os.path.join(fpath, fname)) for fname in sorted(work_dir, key=key_func)])
    output = jnp.sqrt(jnp.where(measurements < 0, 0, measurements))

    return output


def save_tiff(cmplx_img, save_dir):
    """Save provided complex image to specified directory.

    Args:
        cmplx_img (jax.numpy.ndarray): Complex image.
        save_dir (str): Specified directory for saving the image.
    """
    # Ensure the image is a NumPy array
    img = np.array(cmplx_img)

    # Stack the components along a new axis to create a multi-channel image
    img_array = np.stack([np.real(img), np.imag(img), np.abs(img), np.angle(img)])

    # Save the image as a TIFF file with multiple channels
    tiff.imwrite(save_dir, img_array, imagej=True, photometric='minisblack')


def save_array(arr, save_dir):
    """Save an array or list to a specified directory.

    Args:
        arr (numpy.ndarray or list): Numpy array or list to be saved.
        save_dir (str): Directory for saving the array.
    """
    f = open(save_dir, "wb")
    np.save(f, arr)
    f.close


@jit
def fresnel_propagation(field, wavelength, distance, dx):
    """Perform Fresnel propagation of a wavefront from a source plane to an observation plane.

    Args:
        field (jax.numpy.ndarray): The complex wavefront at the source plane, represented as a 2D array.
        wavelength (float): The wavelength of the wave in the same units as the distance and dx.
        distance (float): The propagation distance from the source plane to the observation plane.
        dx (float): The sampling interval in the source plane, i.e., the distance between adjacent points.

    Returns:
        jax.numpy.ndarray: A 2D array representing the complex wavefront at the observation plane.
    """
    # Number of points in each dimension
    N = field.shape[0]

    # Spatial frequency coordinates
    fx = jnp.fft.fftfreq(N, d=dx)
    fy = jnp.fft.fftfreq(N, d=dx)
    FX, FY = jnp.meshgrid(fx, fy, indexing='ij')

    # Quadratic phase factor for Fresnel propagation (Fresnel kernel in the frequency domain)
    H = jnp.exp(-1j * jnp.pi * wavelength * distance * (FX**2 + FY**2))

    # Perform Fourier transform of the source field, apply the Fresnel kernel, and then inverse Fourier transform
    output = jnp.fft.ifft2(jnp.fft.fft2(field) * H)

    return output


def gen_init_probe(y_meas, patch_crds, ref_obj, lpf_sigma=1,
                   fres_propagation=False, wavelength=None, distance=None, dx=None):
    """Generate an initial complex probe based on measured intensity data and reference object.

    Args:
        y_meas (jax.numpy.ndarray): Measured intensity data as a 2D array.
        patch_crds (jax.numpy.ndarray): Coordinates indicating the position of the patch in the reference object.
        ref_obj (jax.numpy.ndarray): Reference object used for initialization.
        lpf_sigma (float, optional): Standard deviation for the Gaussian Low Pass Filter. Defaults to 2.
        fres_propagation (bool, optional): Whether to perform Fresnel propagation on the initialized probe. Defaults to False.
        wavelength (float, optional): Wavelength of the wave, required if fres_propagation is True.
        distance (float, optional): Propagation distance, required if fres_propagation is True.
        dx (float, optional): Sampling interval in the source plane, calculated if not provided.

    Returns:
        jax.numpy.ndarray: The initialized and optionally propagated and filtered complex probe as a 2D array.
    """
    # Ensure wavelength, distance, and dx are provided for Fresnel propagation
    if fres_propagation and (wavelength is None or distance is None):
        raise ValueError("Wavelength and distance must be provided for Fresnel propagation.")

    # Initialization
    k0 = 2 * jnp.pi / wavelength
    Nx, Ny = y_meas.shape[-2], y_meas.shape[-1]
    if dx is None:
        dx = jnp.sqrt(2 * jnp.pi * distance / (k0 * Nx))

    # Extract the relevant patch from the reference object
    patch = img2patch(ref_obj, patch_crds)
    
    # Compute initial probe
    init_probe = jnp.mean(divide_cmplx_numbers(compute_ift(y_meas), patch), axis=0)

    # Perform Fresnel propagation if specified
    if fres_propagation:
        init_probe = fresnel_propagation(init_probe, wavelength, distance, dx)

    # Apply Gaussian Low Pass Filter to both real and imaginary parts
    if lpf_sigma > 0:
        filtered_real = gaussian_filter(jnp.real(init_probe), sigma=lpf_sigma)
        filtered_imag = gaussian_filter(jnp.imag(init_probe), sigma=lpf_sigma)
        output = filtered_real + 1j * filtered_imag
    else:
        output = init_probe

    return output.astype(jnp.complex64)


def gen_init_obj(y_meas, patch_crds, img_sz, ref_probe, lpf_sigma=1):
    """Formulate an initial guess of a complex object for reconstruction.

    Args:
        y_meas (jax.numpy.ndarray): Pre-processed intensity measurements, shape (num_patches, m, n).
        patch_crds (jax.numpy.ndarray): Coordinates of projections, shape (num_patches, 2).
        img_sz (tuple): Size of the full complex image (rows, columns).
        ref_probe (jax.numpy.ndarray): Known or estimated complex probe function, shape (m, n).
        lpf_sigma (float): Standard deviation of the Gaussian kernel for low-pass filtering the initialized guess.

    Returns:
        jax.numpy.ndarray: The initialized guess of the object transmittance image, shape (rows, cols) of img_sz.
    """
    # Calculate RMS of patches based on intensity measurements and the reference probe
    patch_rms = jnp.sqrt(jnp.linalg.norm(y_meas, axis=tuple([-2, -1])) / jnp.linalg.norm(ref_probe))

    # Construct array of patches and reshape it to match the size of the reference probe
    patch_arr = jnp.tile(patch_rms, (ref_probe.shape[0], ref_probe.shape[1], 1))

    # Convert dimensions of array to (num_patch, m, n)
    img_patch = jnp.transpose(patch_arr, (2, 0, 1))

    # Project patches to compose full-sized image with proper weights
    img_wgt = patch2img(jnp.ones_like(y_meas), patch_crds, img_sz=img_sz)
    init_obj = patch2img(img_patch, patch_crds, img_sz=img_sz, weight=img_wgt)

    # Apply LPF to remove high frequencies
    output = gaussian_filter(jnp.abs(init_obj), sigma=lpf_sigma)

    return output.astype(jnp.complex64)


@jit
def compute_nrmse(input_img, ref_img, cstr=None):
    """Compute Normalized Root Mean Square Error (NRMSE) between two images.

    Args:
        input_img (jax.numpy.ndarray): Complex-valued input image for comparison.
        ref_img (jax.numpy.ndarray): Complex-valued reference image for comparison.
        cstr (jax.numpy.ndarray, optional): Constraint area for comparison. If provided, only this region will be considered.

    Returns:
        float: The computed NRMSE between the two images.
    """
    img_rgn = input_img if cstr is None else cstr * input_img
    ref_rgn = ref_img if cstr is None else cstr * ref_img

    num_px = jnp.sum(jnp.abs(cstr)) if cstr is not None else ref_img.size
    mse = jnp.sum(jnp.abs(img_rgn - ref_rgn) ** 2) / num_px

    nrmse = jnp.sqrt(mse) / jnp.sqrt((jnp.sum(jnp.abs(ref_rgn) ** 2)) / num_px)

    return nrmse


@jit
def pha_err(img, ref_img):
    """Calculate the phase error between two complex images.

    Args:
        img (jax.numpy.ndarray): Complex-valued input image.
        ref_img (jax.numpy.ndarray): Complex-valued reference image.

    Returns:
        jax.numpy.ndarray: The phase error for each corresponding element in the images.
    """
    ang_diff = jnp.angle(ref_img) - jnp.angle(img)
    pha_err = jnp.mod(ang_diff + jnp.pi, 2 * jnp.pi) - jnp.pi
    
    return pha_err


@jit
def phase_norm(img, ref_img, cstr=None):
    """Perform phase normalization on a reconstructed complex image.

    Args:
        img (jax.numpy.ndarray): The reconstructed complex image that needs phase normalization.
        ref_img (jax.numpy.ndarray): The known ground truth or reference complex image.
        cstr (jax.numpy.ndarray, optional): Constraint area for comparison. If provided, only this region will be considered.

    Returns:
        jax.numpy.ndarray: The phase-normalized reconstruction.
    """
    img_rgn = img if cstr is None else cstr * img
    ref_rgn = ref_img if cstr is None else cstr * ref_img

    cmplx_scaler = jnp.sum(jnp.conj(img_rgn) * ref_rgn) / (jnp.linalg.norm(img_rgn) ** 2)
    output = cmplx_scaler * img_rgn

    return output.astype(jnp.complex64)