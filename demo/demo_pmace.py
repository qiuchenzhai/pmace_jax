import argparse
import os
import datetime as dt
from shutil import copyfile
import yaml
import numpy as np
import pandas as pd

from pmace_jax.utils import *
from pmace_jax.pmace import *
from demo_utils import *


'''
This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:
 * Downloading demo dataset from specified urls;
 * Loading reference object transmittance image and reference probe profile function;
 * Loading scan locations, simulated measurements, and reconstruction parameters;
 * Computing a reconstruction from the loaded data using PMACE;
 * Displaying and saving the results.
'''
print('This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:\
\n\t * Downloading demo dataset from specified urls; \
\n\t * Loading reference object transmittance image and reference probe profile function; \
\n\t * Loading scan locations, simulated measurements, and reconstruction parameters; \
\n\t * Computing a reconstruction from the loaded data using PMACE; \
\n\t * Displaying and saving the results.\n')


def build_parser():
    parser = argparse.ArgumentParser(description='PMACE demo for ptychographic image reconstruction.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', 
                        const='demo_pmace.yaml',
                        default='config/demo_pmace.yaml')
    return parser


def main():
    # Load config file and parse arguments
    parser = build_parser()
    args = parser.parse_args()
    print("Passing arguments ...")
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # URLs and paths
    dataset_url = config['dataset']['download_url']
    dataset_dir = config['dataset']['save_dir']
    obj_dir = os.path.join(dataset_dir, config['data']['obj_dir'])
    probe_dir = os.path.join(dataset_dir, config['data']['probe_dir'])
    data_dir = os.path.join(dataset_dir, config['data']['data_dir'])
    window_coords = config['data']['window_coords']
    save_dir = config['recon']['out_dir']
    
    # Download dataset
    dataset_path = download_and_extract(dataset_url, dataset_dir)

    # Check and create the output directory
    print("Creating output directory '%s' ..." % save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Load reference images
    print("Loading data ...")
    ref_obj = load_img(obj_dir)
    ref_probe = load_img(probe_dir)

    # Load measurements (diffraction patterns) and pre-process data
    y_meas = load_measurement(os.path.join(data_dir, 'frame_data/'))

    # Load scan positions
    scan_loc_file = pd.read_csv(os.path.join(data_dir, 'Translations.tsv.txt'), sep=None, engine='python', header=0)
    scan_loc = jnp.array(scan_loc_file[['FCx', 'FCy']].to_numpy())

    # Calculate coordinates of projections from scan positions
    patch_crds = get_proj_coords_from_data(scan_loc, y_meas)

    # Formulate initial guess of complex object for reconstruction
    init_obj = gen_init_obj(y_meas, patch_crds, ref_obj.shape, ref_probe=ref_probe)

    # Calculate the coordinates of the reconstruction window
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords
    else:
        xmin, xmax, ymin, ymax = jnp.amin(scan_loc[:, 0]), jnp.amax(scan_loc[:, 0]), jnp.amin(scan_loc[:, 1]), jnp.amax(scan_loc[:, 1])
        
    # Create a reconstruction window using JAX operations
    recon_win = jnp.zeros(ref_obj.shape)

    # JAX arrays are immutable, so we use index_update for in-place updates
    recon_win = recon_win.at[xmin:xmax, ymin:ymax].set(1)

    # Reconstruction parameters
    num_iter = config['recon']['num_iter']
    joint_recon = config['recon']['joint_recon']
    alpha = config['recon']['data_fit_param']
    fig_args = dict(display_win=recon_win, save_fname=os.path.join(save_dir, 'PMACE_recon_cmplx_img'))
    
    # PMACE reconstruction
    pmace_obj = PMACE(pmace_recon, y_meas, patch_crds, init_obj, ref_object=ref_obj, ref_probe=ref_probe,
                      num_iter=num_iter, joint_recon=False, object_data_fit_param=alpha, recon_win=recon_win, save_dir=save_dir)
    pmace_result = pmace_obj()
    plot_synthetic_img(pmace_result['object'], img_title='PMACE', **fig_args)


if __name__ == '__main__':
    main()