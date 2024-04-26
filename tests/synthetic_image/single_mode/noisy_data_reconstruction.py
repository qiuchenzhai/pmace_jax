import argparse
import os
import datetime as dt
from shutil import copyfile
import yaml
import numpy as np
import pandas as pd

from pmace_jax.utils import *
from pmace_jax.pmace import *
from exp_funcs import *


'''
This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:
 * Loading reference object transmittance image and reference probe profile function;
 * Loading scan locations, simulated measurements, and reconstruction parameters;
 * Computing a reconstruction from the loaded data using PMACE;
 * Saving and/or displaying the results.
'''
print('This script demonstrates reconstruction of complex transmittance image using PMACE. Demo functionality includes:\
\n\t * Loading reference object transmittance image and reference probe profile function; \
\n\t * Loading scan locations, simulated measurements, and reconstruction parameters; \
\n\t * Computing a reconstruction from the loaded data using PMACE; \
\n\t * Saving and/or displaying the results.\n')


def build_parser():
    parser = argparse.ArgumentParser(description='Blind ptychographic image reconstruction.')
    parser.add_argument('config_dir', type=str, help='Path to config file.', nargs='?', 
                        const='noisy_data_reconstruction.yaml',
                        default='config/noisy_data_reconstruction.yaml')
    return parser


def main():
    # Load config file and parse arguments
    parser = build_parser()
    args = parser.parse_args()
    print("Passing arguments ...")
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # Objective paths
    data_dir = config['data']['data_dir']
    obj_dir = os.path.join(data_dir, config['data']['ref_object_dir'])
    probe_dir = os.path.join(data_dir, config['data']['ref_probe_dir'])
    meas_dir = os.path.join(data_dir, config['data']['meas_dir'])
    sample_pos_dir = os.path.join(data_dir, config['data']['sample_pos_dir'])
    window_coords = config['data']['window_coords']
    
    # Create the output directory
    save_dir = setup_output_directory_with_timestamp(config['output']['out_dir'])

    # Load reference images
    print("Loading data ...")
    ref_object = load_img(obj_dir)
    ref_probe = load_img(probe_dir)
    
    # Calculate the coordinates of the reconstruction window
    if window_coords is not None:
        xmin, xmax, ymin, ymax = window_coords
    else:
        xmin, xmax, ymin, ymax = jnp.amin(scan_loc[:, 0]), jnp.amax(scan_loc[:, 0]), jnp.amin(scan_loc[:, 1]), jnp.amax(scan_loc[:, 1])
        
    # Create a reconstruction window using JAX operations
    recon_win = jnp.zeros(ref_object.shape)

    # JAX arrays are immutable, so we use index_update for in-place updates
    recon_win = recon_win.at[xmin:xmax, ymin:ymax].set(1)
    
    # Save initial guesses
    save_tiff(ref_object, os.path.join(save_dir, 'ref_object.tiff'))
    save_tiff(ref_probe, os.path.join(save_dir, 'ref_probe.tiff'))
    plot_synthetic_img(ref_object, ref_img=ref_object, display_win=recon_win, save_dir=save_dir + 'ref_object.png')
    plot_recon_probe(ref_probe, ref_img=ref_probe, save_dir=save_dir + 'ref_probe.png')
    
    # Load measurements (diffraction patterns) and pre-process data
    y_meas = load_measurement(meas_dir)
    
    # Use 2D tukey window to suppress background noise
    tukey_win = gen_tukey_2D_window(np.zeros_like(y_meas[0]))
    y_meas = y_meas * tukey_win

    # Load scan positions
    scan_loc_file = pd.read_csv(sample_pos_dir, sep=None, engine='python', header=0)
    scan_loc = jnp.array(scan_loc_file[['FCx', 'FCy']].to_numpy())

    # Save init translations to a CSV file
    df = pd.DataFrame({'FCx': scan_loc[:, 0], 'FCy': scan_loc[:, 1]})
    df.to_csv(os.path.join(save_dir, 'init_Translations.tsv.txt'))
    
    # Calculate coordinates of projections from scan positions
    patch_crds = get_proj_coords_from_data(scan_loc, y_meas)

    # Formulate initial guess of complex object for reconstruction
    fres_prop = config['initialization']['fresnel_propagation']
    source_wavelength = float(config['initialization']['source_wavelength'])
    propagation_distance = float(config['initialization']['propagation_distance'])
    dx = float(config['initialization']['sampling_interval'])
    init_obj = jnp.ones_like(ref_object, dtype=jnp.complex64)
    init_probe = gen_init_probe(y_meas, patch_crds, init_obj, fres_prop=fres_prop,
                                wavelength=source_wavelength, distance=propagation_distance, dx=dx)
    init_obj = gen_init_obj(y_meas, patch_crds, init_obj.shape, ref_probe=init_probe)
    
    # Save initial guesses
    save_tiff(init_obj, os.path.join(save_dir, 'init_object.tiff'))
    save_tiff(init_probe, os.path.join(save_dir, 'init_probe.tiff'))
    plot_synthetic_img(init_obj, ref_img=ref_object, display_win=recon_win, save_dir=save_dir + 'init_object.png')
    plot_recon_probe(init_probe, ref_img=ref_probe, save_dir=save_dir + 'init_probe.png')

    # Reconstruction parameters
    num_iter = config['recon']['num_iter']
    joint_recon = config['recon']['joint_recon']
    recon_args = dict(init_object=init_obj, init_probe=init_probe, recon_win=recon_win, 
                      ref_object=ref_object, ref_probe=ref_probe,
                      num_iter=num_iter, joint_recon=joint_recon)
    # fig_args = dict(display_win=recon_win, save_fname=os.path.join(save_dir, 'PMACE_recon_cmplx_img'))
    
    # PMACE reconstruction
    object_param = config['PMACE']['object_data_fitting_param']
    probe_param = config['PMACE']['probe_data_fitting_param']
    probe_exp = config['PMACE']['probe_exp']

    pmace_dir = os.path.join(save_dir, 'PMACE/')
    pmace_obj = PMACE(pmace_recon, y_meas, patch_crds, **recon_args,
                      object_data_fit_param=object_param, probe_data_fit_param=probe_param, 
                      probe_exp=probe_exp, save_dir=pmace_dir)
    pmace_result = pmace_obj()
    plot_synthetic_img(pmace_result['object'], ref_img=ref_object, 
                       display_win=recon_win, save_dir=pmace_dir + 'recon_object.png')
    plot_recon_probe(pmace_result['probe'][0], ref_img=ref_probe, save_dir=pmace_dir + 'recon_probe.png')
    plot_convergence_curve(pmace_result['err_obj'], pmace_result['err_probe'], pmace_result['err_meas'], 
                           save_dir=pmace_dir + 'convergence_plot.png')
    
    # Save the configuration file to the output directory
    copyfile(args.config_dir, os.path.join(save_dir, 'config.yaml'))


if __name__ == '__main__':
    main()