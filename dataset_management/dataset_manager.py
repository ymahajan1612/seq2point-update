import pandas as pd
import os
import json

class DatasetManager:
    def __init__(self, data_directory, save_path, dataset, appliance_name, debug=False, max_num_houses = None, max_num_rows = 1 * (10**6)):
        self.debug = debug
        self.data_directory = data_directory
        self.save_path = save_path
        self.dataset = dataset.lower()
        self.appliance_name = appliance_name.lower()
        self.appliance_name_formatted = self.appliance_name.replace(" ", "_") 
        self.max_num_houses = max_num_houses

        self.houses = self.getHouses()
        if not self.houses:
            raise ValueError(f"{self.appliance_name} not found in dataset {self.dataset}")

        self.max_num_rows = max_num_rows

        self.house_data_map = self.loadData()
    
    def getHouses(self):
        """
        Get houses for a given dataset and appliance.
        """
        appliance_mappings_dir = os.path.join("dataset_management","data_separation", f"{self.dataset}_appliance_mappings.json")
        houses = []
        with open(appliance_mappings_dir, 'r') as f:
            appliance_mappings = json.load(f)
        for house in appliance_mappings:
            appliance_list = appliance_mappings[house].values()
            # Check if appliance is present in the house
            if self.appliance_name_formatted in appliance_list:
                house_number = int(house.split(" ")[1])
                houses.append(house_number)
            # Stop if we have reached the maximum number of houses
            if self.max_num_houses:
                if len(houses) == self.max_num_houses:
                    break
        if self.debug:
            print(f"Using houses: {houses} for appliance {self.appliance_name}")
        return houses

    def loadData(self):
        """
        Load data for all houses and create a mapping of house number to DataFrame.
        """
        house_data_map = {}

        for house in self.houses:
            house_dir = f"House_{house}"
            house_path = os.path.join(self.data_directory, house_dir)
            aggregate_file = os.path.join(house_path, f'aggregate_H{house}.h5')
            appliance_file = os.path.join(house_path, f'{self.appliance_name_formatted}_H{house}.h5')

            if not os.path.exists(aggregate_file) or not os.path.exists(appliance_file):
                continue

            # Load data
            aggregate_data = pd.read_hdf(aggregate_file)
            appliance_data = pd.read_hdf(appliance_file)

            aggregate_data['time'] = pd.to_datetime(aggregate_data['time'])

            aggregate_data.set_index('time', inplace=True)
            appliance_data.set_index('time', inplace=True)
            # Merge the aggregate and appliance data on timestamp and resample to 6 seconds
            merged_data = aggregate_data.join(appliance_data, how='outer')
            merged_data.index = pd.to_datetime(merged_data.index)
            merged_data = merged_data.resample('6S').mean().fillna(method='backfill', limit=1)
            merged_data.dropna(inplace=True)
            merged_data.reset_index(inplace=True)
            filtered_data = self.selectBestChunk(merged_data)
            house_data_map[house] = filtered_data
        return house_data_map

    def selectBestChunk(self, df, gap_threshold=300):

        df = df.copy()  
        
        df['time'] = pd.to_datetime(df['time'])

        time_diffs = df['time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        window_size = min(self.max_num_rows, int(len(df)*0.3))
        # Find the index of the largest gap in the data
        rolling_max_gaps = time_diffs.rolling(window=window_size, min_periods=1).max().shift(-window_size + 1)

        average_nonzero_ratio = (df[self.appliance_name_formatted] != 0).mean()

        rolling_nonzero_ratios = (df[self.appliance_name_formatted] != 0).rolling(window=window_size, min_periods=1).mean().shift(-window_size + 1)

        if self.debug:
            print(f"Average non-zero ratio: {average_nonzero_ratio}")
        
        valid_chunks = (rolling_max_gaps < gap_threshold) & (rolling_nonzero_ratios >= average_nonzero_ratio)

        if valid_chunks.any():
            print("Valid chunks found")
            best_chunk_index = valid_chunks.idxmax()
            best_chunk = df.loc[best_chunk_index:best_chunk_index + window_size]
        else:
            print("No valid chunks found")
            best_start_index = rolling_max_gaps.idxmin()
            best_chunk = df.loc[best_start_index:best_start_index + window_size]
 
        return best_chunk

    def createData(self):
        for house in self.houses:
            data = self.house_data_map[house]
            os.makedirs(self.save_path, exist_ok=True)
            output_file = os.path.join(self.save_path, f'{self.appliance_name_formatted}_H{house}.csv')
            data.to_csv(output_file, index=False)

        



ukdale_appliances = ["microwave", "dishwasher", "kettle", "washing machine"]
refit_appliances = ["microwave", "dishwasher", "kettle", "washing machine"]
for appliance in refit_appliances:
    appliance_manager = DatasetManager(
        data_directory=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "REFIT_data_separated"),
        save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "REFIT_appliances"),
        dataset='REFIT',
        appliance_name=appliance,
        debug=True,
        max_num_rows=1000000,
    )
    appliance_manager.createData()

