import argparse
import json
import pandas as pd
import os
import nilmtk  

class DataSeparator:
    def __init__(self, file_path, save_path, num_houses, appliance_name=None, dataset_type='REFIT', num_rows=None):
        self.file_path = file_path
        self.save_path = save_path
        self.num_houses = num_houses
        self.appliance_name = appliance_name.lower() if appliance_name else None
        self.dataset_type = dataset_type.upper()
        self.num_rows = num_rows
        self.mapping_file = f"{dataset_type.lower()}_appliance_mappings.json"
        self.output_dir = os.path.join(save_path, f'{dataset_type.lower()}_data_separated')
        os.makedirs(self.output_dir, exist_ok=True)

    def load_mappings(self):
        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def process_data(self):
        appliance_mapping = self.load_mappings()
        num_houses_processed = 0

        for house, channels in appliance_mapping.items():
            house_number = house.split(" ")[1]
            for channel, appliance in channels.items():
                if self.appliance_name and self.appliance_name != appliance.lower():
                    continue

                print(f"Processing data for House {house_number}, Appliance: {appliance}")

                if self.dataset_type == 'REFIT':
                    self._process_refit_data(house_number, channel, appliance)
                elif self.dataset_type == 'UKDALE':
                    self._process_ukdale_data(house_number, channel, appliance)

            num_houses_processed += 1
            if num_houses_processed >= self.num_houses:
                break

    def _process_refit_data(self, house_number, channel, appliance):
        refit_file = os.path.join(self.file_path, f"CLEAN_House{house_number}.csv")
        column_name = "_".join(appliance.lower().split(" "))

        if appliance.lower() == 'aggregate':
            data = pd.read_csv(
                refit_file, header=0, names=['time', 'aggregate'],
                usecols=[0, 1], na_filter=False, parse_dates=True, memory_map=True
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
                print("Mains data length: ", len(mains_data))
            data = pd.DataFrame({'time': mains_data.index, 'aggregate': mains_data.values})
        else:
            appliance_data = dataset.buildings[int(house_number)].elec[int(channel)].power_series_all_data()
            if self.num_rows:
                appliance_data = appliance_data[:min(len(appliance_data),self.num_rows)]
                print("Appliance data length: ", len(appliance_data))

            data = pd.DataFrame({'time': appliance_data.index, appliance_column: appliance_data.values})

        self._save_data(house_number, appliance_column, data)
        print(f"Processed data for House {house_number}, Appliance: {appliance} UKDALE")

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
    parser.add_argument('--dataset_type', type=str, choices=['REFIT', 'UKDALE'], required=True, help='Dataset type.')
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
