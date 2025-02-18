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
  - Example folder structure:
ECO/
├── 01_sm_csv/      # Smart meter readings for House 1
├── 01_plugs_csv/   # Plug-level readings for House 1
├── 02_sm_csv/      # Smart meter readings for House 2
├── 02_plugs_csv/   # Plug-level readings for House 2
├── 03_sm_csv/
├── 03_plugs_csv/
├── 04_sm_csv/
├── 04_plugs_csv/
├── 05_sm_csv/
├── 05_plugs_csv/
├── 06_sm_csv/
├── 06_plugs_csv/



