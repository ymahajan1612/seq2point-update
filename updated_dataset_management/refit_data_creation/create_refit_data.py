import argparse
import json
import pandas as pd
import os

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description='Separate REFIT data by appliance.')
    parser.add_argument('file_path', type=str, help='Path to the directory containing clean REFIT data.')
    parser.add_argument('save_path', type=str, help='Directory where separated data will be saved.')
    parser.add_argument('--num_houses', type=int, help='Number of houses to process (optional).', default=20)
    parser.add_argument('--appliance_name', type=str, help='Appliance name to filter (optional).', default=None)

    args = parser.parse_args()

    print(f"Input file path: {args.file_path}")
    if args.appliance_name:
        print(f"Filtering for appliance: {args.appliance_name}")
    print(f"Data will be saved to: {args.save_path}")

    # Call the data processing function
    create_appliance_data(args.file_path, args.save_path, args.num_houses, args.appliance_name)



def create_appliance_data(refit_file_path, save_path,num_houses, appliance_name=None):
    # Define output directory
    output_dir = os.path.join(save_path, 'refit_data_separated')
    os.makedirs(output_dir, exist_ok=True)
    appliance_name = appliance_name.lower() if appliance_name else None

    # Load appliance mappings
    with open("refit_appliance_mappings.json") as f:
        appliance_mapping = json.load(f)
    
    num_houses_processed = 0

    # Loop through houses and appliances
    for house, channels in appliance_mapping.items():
        for channel, appliance in channels.items():
            appliance = appliance.lower()
            # Skip appliances that don't match the filter (if provided)
            if appliance_name and appliance_name.lower() != appliance.lower():
                continue
            # Extract house number from "House X"
            house_number = house.split(" ")[1]

            # Load the REFIT CSV file for the specific house
            refit_file = os.path.join(refit_file_path, f"CLEAN_House{house_number}.csv")

            if appliance == 'Aggregate':
                data = pd.read_csv(
                    refit_file,
                    header=0,
                    names=['time', 'aggregate'],
                    usecols=[0, 1],
                    na_filter=False,
                    parse_dates=True,
                    memory_map=True
                )
            else:
                if len(appliance.split(" ")) > 1:
                    appliance_column = "_".join(appliance.split(" "))
                else:
                    appliance_column = appliance
                # Load the specific appliance data
                data = pd.read_csv(
                    refit_file,
                    header=0,
                    names=['time', appliance_column],
                    usecols=[0, int(channel) + 2],  # Adjusted for correct column selection
                    na_filter=False,
                    parse_dates=True,
                    memory_map=True
                )
            data['time'] = pd.to_datetime(data['time'])

            # Create house-specific directory
            house_dir = os.path.join(output_dir, f"House_{house_number}")
            os.makedirs(house_dir, exist_ok=True)

            # Save the appliance data
            output_file = os.path.join(house_dir, f"{appliance}_H{house_number}.csv")
            data.to_csv(output_file, index=False, header=True)
            print(f"Saved: {output_file}")
        num_houses_processed += 1
        if num_houses and num_houses_processed >= num_houses:
            break
    
if __name__ == '__main__':
    main()
