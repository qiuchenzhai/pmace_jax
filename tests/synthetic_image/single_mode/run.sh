#!/bin/bash


# Get the current date in the format YYYYMMDD
current_date=$(date +'%Y%m%d')

# Run the program and redirect both stdout and stderr to a log file with the date
python noisy_data_reconstruction.py >> ${current_date}_noisy_recon.log 2>&1 &
