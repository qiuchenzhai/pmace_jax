import os
import yaml
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pmace_jax.utils import *


def setup_output_directory_with_timestamp(output_dir):
    """Set up an output directory with a timestamp.

    Args:
        output_dir (str): Output directory without timestamp.

    Returns:
        str: Timestamped output directory.
    """
    # Determine the timestamp for the output directory
    today_date = dt.date.today()
    date_time = dt.datetime.strftime(today_date, '%Y-%m-%d_%H_%M/')
    output_dir_with_timestamp = os.path.join(output_dir, date_time)

    # Create the output directory if it doesn't exist
    try:
        os.makedirs(output_dir_with_timestamp, exist_ok=True)
        print(f"Output directory '{output_dir_with_timestamp}' created successfully")
    except OSError as error:
        print(f"Output directory '{output_dir_with_timestamp}' cannot be created")

    return output_dir_with_timestamp


def plot_reconstruction_results(cmplx_img, ref_img=None, display_win=None, display=False, save_dir=None,
                                mag_vmax=1.6, mag_vmin=0, phase_vmax=np.pi, phase_vmin=-np.pi,
                                real_vmax=1.2, real_vmin=-1.3, imag_vmax=1, imag_vmin=-1.3, err_vmax=0.1, err_vmin=-0.1):
    """Function to plot reconstruction results.

    Args:
        cmplx_img (jax.numpy.ndarray): The complex-valued image to be plotted.
        ref_img (jax.numpy.ndarray): The reference image used for normalization.
        display_win (jax.numpy.ndarray): A window defining the area to display and compare images. Default is None.
        display (bool): Whether to display the plots. Default is False.
        save_dir (str): The directory where the image files will be saved. Default is None.
        mag_vmax, mag_vmin, etc (float): Parameters for plotting.
    """
    # Initialize the display window and define the area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img)

    # Find the indices of non-zero elements in the pre-defined window
    display_indices = np.nonzero(display_win)

    # Calculate the denoising area boundaries
    row_start = max(0, display_indices[0].min())
    row_end = min(display_indices[0].max() + 1, cmplx_img.shape[0])
    col_start = max(0, display_indices[1].min())
    col_end = min(display_indices[1].max() + 1, cmplx_img.shape[1])

    # Normalize complex image using reference image, if available
    ref_img = ref_img[row_start:row_end, col_start:col_end] if ref_img is not None else cmplx_img[row_start:row_end, col_start:col_end]
    cmplx_img = phase_norm(cmplx_img[row_start:row_end, col_start:col_end], ref_img)

    # Display the amplitude and phase images
    plt.figure(num=None, figsize=(10, 6), dpi=400, facecolor='w', edgecolor='k')

    # ======================================== Plot groundtruth images
    img_title = 'GT'
    # Mag of groundtruth complex image
    plt.subplot(3, 4, 1)
    plt.imshow(np.abs(ref_img), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'mag of {}'.format(img_title))

    # Phase of groundtruth complex image
    plt.subplot(3, 4, 2)
    plt.imshow(np.angle(ref_img), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase of {}'.format(img_title))

    # Real part of groundtruth complex image
    plt.subplot(3, 4, 3)
    plt.imshow(np.real(ref_img), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('real of {}'.format(img_title))

    # Imag part of groundtruth complex image
    plt.subplot(3, 4, 4)
    plt.imshow(np.imag(ref_img), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('imag of {}'.format(img_title))

    # ======================================== Plot reconstructed images
    # Mag of reconstructed complex image
    img_title = 'recon'
    plt.subplot(3, 4, 5)
    plt.imshow(np.abs(cmplx_img), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'mag of {}'.format(img_title))

    # Phase of reconstructed complex image
    plt.subplot(3, 4, 6)
    plt.imshow(np.angle(cmplx_img), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase of {}'.format(img_title))

    # Real part of reconstructed complex image
    plt.subplot(3, 4, 7)
    plt.imshow(np.real(cmplx_img), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('real of {}'.format(img_title))

    # Imag part of reconstructed complex image
    plt.subplot(3, 4, 8)
    plt.imshow(np.imag(cmplx_img), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('imag of {}'.format(img_title))

    # ======================================== Plot error images
    # Magnitude of error image
    plt.subplot(3, 4, 9)
    plt.imshow(np.abs(cmplx_img - ref_img), cmap='gray', vmax=err_vmax, vmin=0)
    plt.title(r'error - amp')
    plt.colorbar()
    plt.axis('off')

    # Phase difference between complex reconstruction and ground truth image
    ang_err = pha_err(cmplx_img, ref_img)
    plt.subplot(3, 4, 10)
    plt.imshow(ang_err, cmap='gray', vmax=err_vmax, vmin=err_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'phase error')

    # Real part of error image between complex reconstruction and ground truth image
    err = cmplx_img - ref_img
    plt.subplot(3, 4, 11)
    plt.imshow(np.real(err), cmap='gray', vmax=err_vmax, vmin=err_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('err - real')

    # Image part of error between complex reconstruction and ground truth image
    plt.subplot(3, 4, 12)
    plt.imshow(np.imag(err), cmap='gray', vmax=err_vmax, vmin=err_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('err - imag')

    if save_dir:
        plt.savefig(save_dir)

    if display:
        plt.show()

    plt.clf()


def plot_synthetic_img(cmplx_img, ref_img=None, display_win=None, display=False, save_dir=None):
    """Function to plot reconstruction results in this experiment.

    Args:
        cmplx_img (jax.numpy.ndarray): The complex-valued image to be plotted.
        ref_img (jax.numpy.ndarray): The reference image used for normalization. Default is None.
        display_win (jax.numpy.ndarray): A window defining the area to display and compare images. Default is None.
        display (bool): Whether to display the plots. Default is False.
        save_dir (str): The directory where the image files will be saved. Default is None.
    """
    # Plot reconstruction results for complex-valued transmittance image  
    plot_reconstruction_results(cmplx_img, ref_img=ref_img, display_win=display_win, display=display, save_dir=save_dir,
                                mag_vmax=1, mag_vmin=0.9, phase_vmax=-0.2, phase_vmin=-1.0, 
                                real_vmax=1, real_vmin=0.5, imag_vmax=-0.2, imag_vmin=-0.8, err_vmax=0.2, err_vmin=-0.2)


def plot_recon_probe(cmplx_img, ref_img=None, display=False, save_dir=None):
    """Function to plot reconstruction results in this experiment.

    Args:
        cmplx_img (jax.numpy.ndarray): The complex-valued image to be plotted.
        ref_img (jax.numpy.ndarray): The reference image used for normalization. Default is None.
        display (bool): Whether to display the plots. Default is False.
        save_dir (str): The directory where the image files will be saved. Default is None.
    """
    # Plot reconstruction results for complex-valued transmittance image  
    plot_reconstruction_results(cmplx_img, ref_img=ref_img, display_win=None, display=display, save_dir=save_dir,
                                mag_vmax=90, mag_vmin=0, phase_vmax=np.pi, phase_vmin=-np.pi, 
                                real_vmax=30, real_vmin=-70, imag_vmax=10, imag_vmin=-60, err_vmax=2, err_vmin=-2)


def plot_convergence_curve(err_obj=None, err_probe=None, err_meas=None, display=False, save_dir=None):
    """Plot the convergence curves for NRMSE in object, probe, and detector planes.

    Args:
        err_obj (list): List of NRMSE values in the object plane.
        err_probe (list): List of NRMSE values in the probe plane.
        err_meas (list): List of NRMSE values in the detector plane.
        display (bool): Whether to display the plots. Default is False.
        save_dir (str): The directory where the image files will be saved. Default is None.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), dpi=100)

    # Plot NRMSE in object plane
    if err_obj:
        axes[0].semilogy(np.arange(0, len(err_obj), 1), err_obj, label='NRMSE_obj')
        axes[0].set_ylabel('NRMSE in object plane (log scale)')
        axes[0].set_xlabel('Number of Iterations')

    # Plot NRMSE in probe plane
    if err_probe:
        for idx, err_mode in enumerate(err_probe):
            axes[1].semilogy(np.arange(len(err_meas) - len(err_mode), len(err_meas), 1), err_mode, label='NRMSE_mode_{}'.format(idx))
        axes[1].set_ylabel('NRMSE in probe plane (log scale)')
        axes[1].set_xlabel('Number of Iterations')

    # Plot NRMSE in detector plane
    if err_meas:
        axes[2].semilogy(np.arange(0, len(err_meas), 1), err_meas, label='NRMSE_meas')
        axes[2].set_ylabel('NRMSE in detector plane (log scale)')
        axes[2].set_xlabel('Number of Iterations')

    # Adjust x-axis ticks
    for ax in axes:
        ax.set_xticks(np.arange(0, len(err_meas), 10))

    # Add legend, grid, and display the plot
    axes[2].legend(loc='best')
    for ax in axes:
        ax.grid(which='both', alpha=0.2)
        ax.grid(which='minor', alpha=0.1)

    if save_dir is not None:
        plt.savefig(save_dir)

    if display:
        plt.show()

    plt.clf()