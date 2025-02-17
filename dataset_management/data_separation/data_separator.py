import json
import pandas as pd
import numpy as np
from pandas import HDFStore
import os
import nilmtk
from datetime import datetime
import re

class DataSeparator:
    """
    Class to separate energy dataset data by appliance for each house.
    """
    def __init__(self, file_path, save_path, num_houses=None, appliance_name=None, dataset_type='REFIT'):
        self.file_path = file_path
        self.save_path = save_path
        self.appliance_name = appliance_name.lower() if appliance_name else None
        self.dataset_type = dataset_type.upper()
        self.num_houses = num_houses

        # Load data ranges for ECO dataset as the range differs for aggegate and appliance data
        self.eco_data_ranges = None
        if self.dataset_type == "ECO":
            date_ranges_file = os.path.join("dataset_management","data_separation","eco_data_ranges.json")
            with open(date_ranges_file, "r") as f:
                self.eco_data_ranges = json.load(f)

        # Load appliance mappings for the dataset
        self.mapping_file = os.path.join("dataset_management","data_separation",f"{dataset_type.lower()}_appliance_mappings.json")
        self.output_dir = os.path.join(save_path, f'{dataset_type.upper()}_data_separated')
        os.makedirs(self.output_dir, exist_ok=True)

    

    def load_mappings(self):
        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def process_data(self):
        appliance_mapping = self.load_mappings()
        print(f"Processing data for dataset: {self.dataset_type}")
        num_houses_processed = 0

        for house, channels in appliance_mapping.items():
            house_number = house.split(" ")[1]
            # special handling for ECO dataset where the aggregate data is stored in a separate directory
            if self.dataset_type == 'ECO':
                self._process_eco_aggregate_data(house_number)
            for channel, appliance in channels.items():
                if self.appliance_name and self.appliance_name != appliance.lower():
                    continue

                print(f"Processing data for House {house_number}, Appliance: {appliance}")

                if self.dataset_type == 'REFIT':
                    self._process_refit_data(house_number, channel, appliance)
                elif self.dataset_type == 'UKDALE':
                    self._process_ukdale_data(house_number, channel, appliance)
                elif self.dataset_type == 'ECO':
                    self._process_eco_appliance_data(house_number, channel, appliance)
                elif self.dataset_type == 'REDD':
                    self._process_redd_data(house_number, channel, appliance)

            num_houses_processed += 1
            if self.num_houses:
                if num_houses_processed >= self.num_houses:
                    break

    def _process_refit_data(self, house_number, channel, appliance):
        refit_file = os.path.join(self.file_path, f"CLEAN_House{house_number}.csv")

        if len(appliance.split(" ")) > 1: 
            column_name = "_".join(appliance.lower().split(" "))
        else:
            column_name = appliance.lower()

        if appliance.lower() == 'aggregate':
            data = pd.read_csv(
                refit_file, header=0, names=['time', 'aggregate'],
                usecols=[0, 2], na_filter=False, parse_dates=True, memory_map=True
            )
        else:
            data = pd.read_csv(
                refit_file, header=0, names=['time', column_name],
                usecols=[0, int(channel) + 2], na_filter=False, parse_dates=True, memory_map=True
            )

        self._save_data(house_number, column_name, data)

    def _process_ukdale_data(self, house_number, channel, appliance):
        appliance_column = "_".join(appliance.lower().split(" "))
        
        key = f'/building{house_number}/elec/meter{channel}'
        data = None
        with HDFStore(os.path.join(self.file_path, "ukdale.h5")) as store:
            df = store.get(key)
            data = pd.DataFrame({'time': df.index, appliance_column: df.values.flatten()})

        data['time'] = data['time'].astype(str).apply(lambda x: x.split('+')[0])
        self._save_data(house_number, appliance_column, data)
    
    def _process_redd_data(self, house_number, channel, appliance):
        appliance_column = "_".join(appliance.lower().split(" "))
        data = None
        dataset = nilmtk.DataSet(os.path.join(self.file_path, "redd.h5"))
        if appliance_column == 'aggregate':
            mains_data1 = dataset.buildings[int(house_number)].elec[1].power_series_all_data()
            mains_df1 = pd.DataFrame({'time': mains_data1.index, 'aggregate_1': mains_data1.values})
            mains_data2 = dataset.buildings[int(house_number)].elec[2].power_series_all_data()
            mains_df2 = pd.DataFrame({'time': mains_data2.index, 'aggregate_2': mains_data2.values})
            mains_df1.set_index('time', inplace=True)
            mains_df2.set_index('time', inplace=True)
            data = mains_df1.join(mains_df2, how='outer')
            data['aggregate'] = data.iloc[:].sum(axis=1)
            data.reset_index(inplace=True)
            data.drop(columns=['aggregate_1', 'aggregate_2'], inplace=True)
        else:
            appliance_data = dataset.buildings[int(house_number)].elec[int(channel)].power_series_all_data()
            data = pd.DataFrame({'time': appliance_data.index, appliance_column: appliance_data.values})
        data.dropna(inplace=True)
        data['time'] = data['time'].astype(str).apply(lambda x: x.rsplit('-',1)[0])
        self._save_data(house_number, appliance_column, data)   
        
        
    def _generate_timestamps(self, file_name):
        num_rows=86400
        base_date = file_name.split(".")[0]
        start_time = pd.to_datetime(base_date)
        return pd.date_range(start=start_time, periods=num_rows, freq='S')
    
    def _process_eco_aggregate_data(self, house_number):
        print(f"Processing aggregate data for ECO, House {house_number}")
        house_dir_formatted = f"0{house_number}"
        smart_meter_dir = os.path.join(self.file_path, f'{house_dir_formatted}_sm_csv', house_dir_formatted)
        all_aggregate_data = []
        for file_name in sorted(os.listdir(smart_meter_dir)):
            if not file_name.endswith('.csv'):
                continue
            match = re.match(r'(\d{4}-\d{2}-\d{2})', file_name)
            if match:
                file_name = match.group(1) + ".csv"
            start_date = datetime.strptime(self.eco_data_ranges[house_number]['start'], "%Y-%m-%d")
            end_date = datetime.strptime(self.eco_data_ranges[house_number]['end'], "%Y-%m-%d")
            file_date = datetime.strptime(file_name.split(".")[0], "%Y-%m-%d")
            if file_date < start_date or file_date > end_date:
                continue

            file_path = os.path.join(smart_meter_dir, file_name)

            aggregate_df = pd.read_csv(file_path, header=None, usecols=[0], names=['aggregate'])
            aggregate_df['time'] = self._generate_timestamps(file_name)
            aggregate_df = aggregate_df[['time', 'aggregate']]
            all_aggregate_data.append(aggregate_df)
        
        aggregate_data = pd.concat(all_aggregate_data, axis=0)
        self._save_data(house_number, 'aggregate', aggregate_data)

    def _process_eco_appliance_data(self, house_number, channel, appliance):
        house_dir_formatted = f"0{house_number}"
        plug_dir = os.path.join(self.file_path, f'{house_dir_formatted}_plugs_csv', house_dir_formatted)    
        if len(appliance.split(" ")) > 1: 
            column_name = "_".join(appliance.lower().split(" "))
        else:
            column_name = appliance.lower()
        plug_subdir = os.path.join(plug_dir, channel)
        if os.path.exists(plug_subdir):
            all_plug_data = []  
            for file_name in sorted(os.listdir(plug_subdir)):
                if not file_name.endswith('.csv'):
                    continue
                match = re.match(r'(\d{4}-\d{2}-\d{2})', file_name)
                if match:
                    file_name = match.group(1) + ".csv"
                start_date = datetime.strptime(self.eco_data_ranges[house_number]['start'], "%Y-%m-%d")
                end_date = datetime.strptime(self.eco_data_ranges[house_number]['end'], "%Y-%m-%d")
                file_date = datetime.strptime(file_name.split(".")[0], "%Y-%m-%d")
                if file_date < start_date or file_date > end_date:
                    continue
                file_path = os.path.join(plug_subdir, file_name)

                plug_df = pd.read_csv(file_path, header=None, usecols=[0], names=[column_name])
                plug_df['time'] = self._generate_timestamps(file_name)
                plug_df = plug_df[['time', column_name]]
                plug_df.replace(-1, np.NaN, inplace=True)

                all_plug_data.append(plug_df)

            if all_plug_data:
                appliance_df = pd.concat(all_plug_data, axis=0)

                self._save_data(house_number, column_name, appliance_df)
            

    def _save_data(self, house_number, appliance_column, data):
        house_dir = os.path.join(self.output_dir, f"House_{house_number}")
        os.makedirs(house_dir, exist_ok=True)

        # Save data to HDF5 file
        output_file = os.path.join(house_dir, f"{appliance_column}_H{house_number}.h5")
        data.to_hdf(output_file, key='dataset', mode='w', format='table')
        print(f"Saved: {output_file}")



data_separator = DataSeparator(
    file_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "CLEAN_REFIT_081116"),
    save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents"),
    appliance_name=None,
    dataset_type="refit",
)
data_separator.process_data()