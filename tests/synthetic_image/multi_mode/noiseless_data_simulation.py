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
This script demonstrates the simulation of noiseless multi-mode (two-mode) ptychographic data. Demo functionality includes:
 * Loading reference object transmittance image and reference probe profile functions;
 * Loading scan locations and simulating noiseless measurements;
 * Saving the simulated intensity data to specified location.
'''
print('This script demonstrates the simulation of noisy ptychographic data. Demo functionality includes:\
\n\t * Loading reference object transmittance image and reference probe profile functions; \
\n\t * Loading scan locations and simulating noiseless measurements; \
\n\t * Saving the simulated intensity data to specified location.\n')


def build_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Noiseless Two-mode Ptychographic Data Simulation')
    parser.add_argument('config_dir', type=str, help='config_dir', nargs='?', 
                        const='config/noiseless_data_simulation.yaml',
                        default='config/noiseless_data_simulation.yaml')
    
    return parser


def main():
    # Load config file and parse arguments
    parser = build_parser()
    args = parser.parse_args()
    print("Passing arguments ...")
    with open(args.config_dir, 'r') as f:
        config = yaml.safe_load(f)

    # Read directories
    gt_object_dir = config['data']['gt_object_dir']
    gt_probe_mode_0_dir = config['data']['gt_probe_0_dir']
    gt_probe_mode_1_dir = config['data']['gt_probe_1_dir']

    # Extract simulation parameters
    num_meas = config['simulation']['num_meas']
    probe_spacing = config['simulation']['probe_spacing']
    max_scan_loc_offset = config['simulation']['max_scan_loc_offset']
    add_noise = config['simulation']['add_noise']
    if add_noise:
        peak_photon_rate = float(config['simulation']['peak_photon_rate'])
        shot_noise_rate = float(config['simulation']['shot_noise_rate'])
    else:
        peak_photon_rate = None
        shot_noise_rate = None
        
    distribute_energy = config['simulation']['distribute_energy']
    mode_orthogonalization = config['simulation']['mode_orthogonalization']
    
    # Create the output directory
    save_dir = config['output']['out_dir']
    print("Creating data directory '%s' ..." % save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the random key
    jnp_cdtype = jnp.complex64
    rng_key = random.PRNGKey(0)
    
    # Load reference images
    print("Reading ground truth images ...")
    ref_object = load_img(gt_object_dir)
    ref_probe_0 = load_img(gt_probe_mode_0_dir)
    ref_probe_1 = load_img(gt_probe_mode_1_dir)
    
    # Adjust the size of image
    crop_image = config['simulation']['crop_image']
    if crop_image:
        print("Cropping image to size {} ...".format(image_size))
        image_size = config['simulation']['image_size']
        gt_object = gt_object[0:image_size[0], 0:image_size[1]]
    
    # Adjust the energy distribution
    if distribute_energy:
        print("Balancing the energy ratio between probe modes ...")
        energy_ratio = config['simulation']['energy_ratio']
        ref_probe_0, ref_probe_1 = adjust_energy_distribution(ref_probe_0, ref_probe_1, energy_ratio)
    else:
        ref_probe_0, ref_probe_1 = gt_probe_mode_0, gt_probe_mode_1
 
    # Orthogonalize the probe mode        
    if mode_orthogonalization:
        print("Orthogonalizing probe modes ...")
        orthogonalized_imgs = orthogonalize_images(jnp.array([ref_probe_0, ref_probe_1]))
        ref_probe_0, ref_probe_1 = orthogonalized_imgs[0], orthogonalized_imgs[1]
        
    # Save initial guesses
    save_tiff(ref_object, os.path.join(save_dir, 'ref_object.tiff'))
    save_tiff(ref_probe_0, os.path.join(save_dir, 'ref_probe_0.tiff'))
    save_tiff(ref_probe_1, os.path.join(save_dir, 'ref_probe_1.tiff'))
    plot_FakeIC_img(ref_object, ref_img=ref_object, save_dir=save_dir + 'ref_object.png')
    plot_mode_0(ref_probe_0, ref_img=ref_probe_0, save_dir=save_dir + 'ref_probe_0.png')
    plot_mode_1(ref_probe_1, ref_img=ref_probe_1, save_dir=save_dir + 'ref_probe_1.png')
        
    # Generate scan positions
    print("Creating and saving scan locations ...")
    scan_loc = gen_scan_loc(ref_object, ref_probe_0, num_meas, probe_spacing,
                            randomization=True, max_offset=5, rng_key=rng_key)
    df = pd.DataFrame({'FCx': scan_loc[:, 0], 'FCy': scan_loc[:, 1]})
    df.to_csv(save_dir + 'Translations.tsv.txt')

    # Initialize measurement array
    y_meas = jnp.zeros((len(scan_loc), ref_probe_0.shape[0], ref_probe_0.shape[1]))
    scan_coords = get_proj_coords_from_data(scan_loc, y_meas)

    # Simulate measurements
    print("Simulating measurements ...")
    intensity_data = gen_syn_data(ref_object, jnp.array([ref_probe_0, ref_probe_1]), scan_coords,
                                  add_noise=add_noise, peak_photon_rate=peak_photon_rate,
                                  shot_noise_pm=shot_noise_rate, save_dir=save_dir + 'frame_data/')
    
    # Save the config file to the output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    copyfile(args.config_dir, save_dir + 'config.yaml')
    
    
if __name__ == '__main__':
    main()