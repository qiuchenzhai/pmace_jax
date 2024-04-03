import os, sys
import urllib.request
import glob
import tarfile
import numpy as np
import matplotlib.pyplot as plt


def imshow_with_options(array, title='', vmin=None, vmax=None, cmap='viridis', show=False):
    """ Display an array as an image along with optional title and intensity window.
    Args:
        array: The array to display
        title: The title for the plot
        vmin: Minimum of the intensity window - same as vmin in imshow
        vmax: Maximum of the intensity window - same as vmax in imshow
        cmap: The color map as in imshow - same as cmap in imshow
        show: If true, then plt.show() is called to display immediately, otherwise call
              fig.show() on the object returned from this function to show the plot.

    Returns:
        The pyplot figure object
    """
    fig = plt.figure(layout="constrained")
    plt.imshow(array, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='none')
    plt.colorbar()
    plt.title(title)
    if show:
        plt.show()
    return fig


def plot_synthetic_img(cmplx_img, img_title='img', ref_img=None, display_win=None, display=False, save_fname=None,
                       fig_sz=[8, 3], mag_vmax=1, mag_vmin=0.5, phase_vmax=0, phase_vmin=-np.pi/4,
                       real_vmax=1.1, real_vmin=0.8, imag_vmax=0, imag_vmin=-0.6):
    """Plot complex object images and error images compared with a reference image.

    Args:
        cmplx_img (numpy.ndarray): Complex-valued image.
        img_title (str): Title for the complex image.
        ref_img (numpy.ndarray or None): Reference image. If provided, error images will be displayed.
        display_win (numpy.ndarray or None): Pre-defined window for displaying images.
        display (bool): Display images if True.
        save_fname (str or None): Save images to the specified file directory.
        fig_sz (list): Size of image plots in inches (width, height).
        mag_vmax (float): Maximum value for showing image magnitude.
        mag_vmin (float): Minimum value for showing image magnitude.
        phase_vmax (float): Maximum value for showing image phase.
        phase_vmin (float): Minimum value for showing image phase.
        real_vmax (float): Maximum value for showing the real part of the image.
        real_vmin (float): Minimum value for showing the real part of the image.
        imag_vmax (float): Maximum value for showing the imaginary part of the image.
        imag_vmin (float): Minimum value for showing the imaginary part of the image.
    """
    # Plot error images if a reference image is provided
    show_err_img = False if (ref_img is None) or (np.linalg.norm(cmplx_img - ref_img) < 1e-9) else True

    # Initialize the window and determine the area for showing and comparing images
    if display_win is None:
        display_win = np.ones_like(cmplx_img, dtype=np.complex64)
    non_zero_idx = np.nonzero(display_win)
    blk_idx = [np.amin(non_zero_idx[0]), np.amax(non_zero_idx[0])+1, np.amin(non_zero_idx[1]), np.amax(non_zero_idx[1])+1]
    cmplx_img_rgn = cmplx_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    if ref_img is not None:
        ref_img_rgn = ref_img[blk_idx[0]:blk_idx[1], blk_idx[2]:blk_idx[3]]
    
    # Display the amplitude and phase images
    plt.figure(num=None, figsize=(fig_sz[0], fig_sz[1]), dpi=400, facecolor='w', edgecolor='k')
    
    # Magnitude of the reconstructed complex image
    plt.subplot(2, 4, 1)
    plt.imshow(np.abs(cmplx_img_rgn), cmap='gray', vmax=mag_vmax, vmin=mag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'Magnitude of {}'.format(img_title))
                                  
    # Phase of the reconstructed complex image
    plt.subplot(2, 4, 2)
    plt.imshow(np.angle(cmplx_img_rgn), cmap='gray', vmax=phase_vmax, vmin=phase_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title(r'Phase of {}'.format(img_title))
    
    # Real part of the reconstructed complex image
    plt.subplot(2, 4, 3)
    plt.imshow(np.real(cmplx_img_rgn), cmap='gray', vmax=real_vmax, vmin=real_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('Real Part of {}'.format(img_title))
    
    # Imaginary part of the reconstructed complex image
    plt.subplot(2, 4, 4)
    plt.imshow(np.imag(cmplx_img_rgn), cmap='gray', vmax=imag_vmax, vmin=imag_vmin)
    plt.colorbar()
    plt.axis('off')
    plt.title('Imaginary Part of {}'.format(img_title))

    if show_err_img:
        # Amplitude of the difference between complex reconstruction and ground truth image
        plt.subplot(2, 4, 5)
        plt.imshow(np.abs(cmplx_img_rgn - ref_img_rgn), cmap='gray', vmax=0.2, vmin=0)
        plt.title(r'Error - Amp')
        plt.colorbar()
        plt.axis('off')
        
        # Phase difference between complex reconstruction and ground truth image
        ang_err = pha_err(cmplx_img_rgn, ref_img_rgn)
        plt.subplot(2, 4, 6)
        plt.imshow(ang_err, cmap='gray', vmax=np.pi/2, vmin=-np.pi/2)
        plt.colorbar()
        plt.axis('off')
        plt.title('Phase Error')
        
        # Real part of the error image between complex reconstruction and ground truth image
        err = cmplx_img_rgn - ref_img_rgn
        plt.subplot(2, 4, 7)
        plt.imshow(np.real(err), cmap='gray', vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.axis('off')
        plt.title('Error - Real')
        
        # Imaginary part of the error between complex reconstruction and ground truth image
        plt.subplot(2, 4, 8)
        plt.imshow(np.imag(err), cmap='gray', vmax=0.2, vmin=-0.2)
        plt.colorbar()
        plt.axis('off')
        plt.title('Error - Imag')

    if save_fname is not None:
        plt.savefig('{}.png'.format(save_fname))
    if display:
        plt.show()
    plt.clf()
    
    
def download_and_extract(download_url, save_dir):
    """Download the file from ``download_url``, and save the file to ``save_dir``. 
    
    If the file already exists in ``save_dir``, user will be queried whether it is desired to download and overwrite the existing files.
    If the downloaded file is a tarball, then it will be extracted before saving. 
    Code reference: `https://github.com/cabouman/mbircone/`
    
    Args:
        download_url: url to download the data. This url needs to be public.
        save_dir (string): directory where downloaded file will be saved. 
        
    Returns:
        path to downloaded file. This will be ``save_dir``+ downloaded_file_name 
    """
    is_download = True
    local_file_name = download_url.split('/')[-1]
    save_path = os.path.join(save_dir, local_file_name)
    if os.path.exists(save_path):
        is_download = query_yes_no(f"{save_path} already exists. Do you still want to download and overwrite the file?")
    if is_download:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Download the data from URL.
        print("Downloading file ...")
        try:
            urllib.request.urlretrieve(download_url, save_path)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL authentication failed! Currently we do not support downloading data from a url that requires authentication.')
            elif e.code == 403:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL forbidden! Please make sure the provided URL is public.')
            elif e.code == 404:
                raise RuntimeError(
                    f'HTTP status code {e.code}: URL not Found! Please check and make sure the download URL provided is correct.')
            else:
                raise RuntimeError(
                    f'HTTP status code {e.code}: {e.reason}. For more details please refer to https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        except urllib.error.URLError as e:
            raise RuntimeError('URLError raised! Please check your internet connection.')
        print(f"Download successful! File saved to {save_path}")
    else:
        print("Skipped data download and extraction step.")
    # Extract the downloaded file if it is tarball
    if save_path.endswith(('.tar', '.tar.gz', '.tgz')):
        if is_download:
            tar_file = tarfile.open(save_path)
            print(f"Extracting tarball file to {save_dir} ...")
            # Extract to save_dir.
            tar_file.extractall(save_dir)
            tar_file.close
            print(f"Extraction successful! File extracted to {save_dir}")
        save_path = save_dir
        # Remove invisible files with "._" prefix 
        for spurious_file in glob.glob(save_dir + "/**/._*", recursive=True):
            os.remove(spurious_file)
    # Parse extracted dir and extract data if necessary
    return save_path


def query_yes_no(question, default="n"):
    """Ask a yes/no question via input() and return the answer.
    
    Code reference: `https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input/3041990`.
        
    Args:
        question (string): Question that is presented to the user.
        
    Returns:
        Boolean value: True for "yes" or "Enter", or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = f" [y/n, default={default}] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
    return