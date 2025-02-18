# Sequence-to-point (seq2point) Toolkit non-intrusive load monitoring (NILM)


This repository contains an implementation of Seq2Point models for Non-Intrusive Load Monitoring (NILM), based on the approach introduced in the paper:

Mingjun Zhong, Nigel Goddard, Stephen Sutton, and Charles Gillan (2018).
"Sequence-to-Point Learning with Neural Networks for Non-Intrusive Load Monitoring"

The code has been adapted from the original implementation by Mingjun Zhong: https://github.com/MingjunZhong/seq2point-nilm.

The project aims to provide a toolkit that standardizes benchmarking for the latest seq2point architectures across various energy datasets, providing performance metrics (mean absolute error, signal aggregate error and inference time) in an easy to compare format across multiple datasets such as REDD, REFIT, ECO and UKDALE. 

## Overview of Key Modules

### data_separator.py
The data separator module is designed to disaggregate appliance-specific power consumption data from multiple NILM (Non-Intrusive Load Monitoring) datasets. The currently supported datasets and their expected formats is as follows:
- UKDALE: HDF5 file (https://data.ukedc.rl.ac.uk/cgi-bin/data_browser/browse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-FULL-disaggregated/ukdale.h5.zip)
- REFIT: Cleaned format (https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned)
- REDD: .HDF5 file (https://tokhub.github.io/dbecd/links/redd.html)
- ECO: (Requires pre-processing) Before using the ECO dataset, some preprocessing is required. The dataset is originally available as **zipped files**, where each **house** has separate ZIP archives for:
  - **Smart meter (aggregate) data**
  - **Plug-level appliance data** (one zip file per plug)
  - Unzip the smart meter and plug level appliance data to a single folder (CSV version)

The following code processes the data to the required format: 

data_separator = DataSeparator(
    file_path=*file location of the raw data*,
    save_path=*save path for the files*,
    appliance_name=...,
    dataset_type=...,
)
data_separator.process_data()

Running this saves the data in HDF5 format

### dataset_manager.py
 responsible for loading, processing, and preparing appliance-specific data from NILM datasets for training Seq2Point models. It handles data selection, resampling, normalization, and saving in a structured format.
 
It extracts aggregate power readings and individual appliance consumption, ensuring consistent sampling intervals and selects a chunk of data with minimal gaps and sufficient appliance activity. 

Saves the data for a specified appliance in a dataset as a csv with filename: [appliance]_H[house number].csv

Also creates a normalisation parameters .json file containing the aggregate and appliance mean and standard deviation 

Example CSV file from UKDALE dataset (dishwasher_H1.csv):
  time, aggregate, dishwasher
  2013-05-17 09:35:12, 0.13711391113313803, -0.07084559614012528
  2013-05-17 09:35:18, 0.134153645793339, -0.07084559614012528
  2013-05-17 09:35:24, 0.1430344418127361, -0.07084559614012528
  2013-05-17 09:35:30, 0.1430344418127361, -0.07084559614012528

### train_model.py 

The module for training a seq2point model. Class parameters are as follows:
model_name, train_csv_dirs, validation_csv_dirs, appliance, dataset, model_save_dir, window_length 
  model_name (str): Name of the model to train.
  train_csv_dirs (list): List of file paths to the training CSVs.
  validation_csv_dirs (list): list of file paths to the validation CSVs
  appliance (str): Name of the appliance to train the model for.
  dataset (str): Name of the dataset.
  model_save_dir (str): Directory to save the trained model.
  window_length (int): Length of the input window.

### test_model.py 

The module for training seq2point models. Class parameters are as follows:
        model_name (str): Name of the model to test.
        model_state_dir (str): Directory to load the model state from.
        test_csv_dir (str): Directory to load the test CSV from.
        appliance (str): Name of the appliance to test the model for.
        normalisation_parameters_dir (str): Directory to load the normalisation parameters from.

