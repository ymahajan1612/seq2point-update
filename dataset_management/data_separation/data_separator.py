import argparse
import json
import pandas as pd
import os
import nilmtk  
from datetime import datetime
import re

class DataSeparator:
    """
    Class to separate energy dataset data by appliance for each house.
    """
    def __init__(self, file_path, save_path, num_houses, appliance_name=None, dataset_type='REFIT', num_rows=None):
        self.file_path = file_path
        self.save_path = save_path
        self.num_houses = num_houses
        self.appliance_name = appliance_name.lower() if appliance_name else None
        self.dataset_type = dataset_type.upper()
        self.num_rows = num_rows

        # Load data ranges for ECO dataset as the range differs for aggegate and appliance data
        self.eco_data_ranges = None
        if self.dataset_type == "ECO":
            with open("eco_data_ranges.json", "r") as f:
                self.eco_data_ranges = json.load(f)

        # Load appliance mappings for the dataset
        self.mapping_file = f"{dataset_type.lower()}_appliance_mappings.json"
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

            num_houses_processed += 1
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
            if self.num_rows:
                data = data.iloc[:min(len(data),self.num_rows)]
        else:
            data = pd.read_csv(
                refit_file, header=0, names=['time', column_name],
                usecols=[0, int(channel) + 2], na_filter=False, parse_dates=True, memory_map=True
            )
            if self.num_rows:
                data = data.iloc[:min(len(data),self.num_rows)]

        self._save_data(house_number, column_name, data)

    def _process_ukdale_data(self, house_number, channel, appliance):
        dataset = nilmtk.DataSet(os.path.join(self.file_path, "ukdale.h5"))
        appliance_column = "_".join(appliance.lower().split(" "))

        if appliance.lower() == 'aggregate':
            mains_data = dataset.buildings[int(house_number)].elec.mains().power_series_all_data()
            if self.num_rows:
                mains_data = mains_data[:min(len(mains_data),self.num_rows)]
            mains_data.index = mains_data.index.split('+')[0]
            data = pd.DataFrame({'time': mains_data.index, 'aggregate': mains_data.values})
        else:
            appliance_data = dataset.buildings[int(house_number)].elec[int(channel)].power_series_all_data()
            if self.num_rows:
                appliance_data = appliance_data[:min(len(appliance_data),self.num_rows)]

            data = pd.DataFrame({'time': appliance_data.index, appliance_column: appliance_data.values})

        self._save_data(house_number, appliance_column, data)
    
    def _generate_timestamps(self, file_name, num_rows=86400):
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
        if self.num_rows:
            aggregate_data = aggregate_data.iloc[:min(len(aggregate_data),self.num_rows)]
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

                all_plug_data.append(plug_df)

            if all_plug_data:
                appliance_df = pd.concat(all_plug_data, axis=0)
                if self.num_rows:
                    appliance_df = appliance_df.iloc[:min(len(appliance_df),self.num_rows)]

                self._save_data(house_number, column_name, appliance_df)
            

    def _save_data(self, house_number, appliance_column, data):
        house_dir = os.path.join(self.output_dir, f"House_{house_number}")
        os.makedirs(house_dir, exist_ok=True)
        output_file = os.path.join(house_dir, f"{appliance_column}_H{house_number}.csv")
        data.to_csv(output_file, index=False, header=True)
        print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Separate energy dataset data by appliance.')
    parser.add_argument('file_path', type=str, help='Path to the dataset directory.')
    parser.add_argument('save_path', type=str, help='Directory to save separated data.')
    parser.add_argument('--num_houses', type=int, default=20, help='Number of houses to process (default: 20).')
    parser.add_argument('--appliance_name', type=str, default=None, help='Filter by specific appliance name (optional).')
    parser.add_argument('--dataset_type', type=str, choices=['REFIT', 'UKDALE', 'ECO'], required=True, help='Dataset type.')
    parser.add_argument('--num_rows', type=int, default=None, help='Number of rows of data to process (optional).')

    args = parser.parse_args()
    print(f"Processing {args.dataset_type} dataset...")
    print(f"Input file path: {args.file_path}")
    print(f"Data will be saved to: {args.save_path}")

    separator = DataSeparator(
        file_path=args.file_path,
        save_path=args.save_path,
        num_houses=args.num_houses,
        appliance_name=args.appliance_name,
        dataset_type=args.dataset_type,
        num_rows = args.num_rows
    )
    separator.process_data()

if __name__ == "__main__":
    main()
