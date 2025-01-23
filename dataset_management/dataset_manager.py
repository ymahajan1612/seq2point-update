import pandas as pd
import os
import json

class DatasetManager:
    def __init__(self, data_directory, save_path, dataset, appliance_name, debug=False, max_num_houses = 4, num_rows = 1 * (10**6)):
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

        self.num_rows = num_rows

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
            if self.appliance_name_formatted in appliance_list:
                house_number = int(house.split(" ")[1])
                houses.append(house_number)
            if len(houses) == self.max_num_houses:
                break
        if self.debug:
            print(f"Using houses: {houses}")
        return houses

    def loadData(self):
        """
        Load data for all houses and create a mapping of house number to DataFrame.
        """
        house_data_map = {}

        for house in self.houses:
            house_dir = f"House_{house}"
            house_path = os.path.join(self.data_directory, house_dir)
            aggregate_file = os.path.join(house_path, f'aggregate_H{house}.csv')
            appliance_file = os.path.join(house_path, f'{self.appliance_name_formatted}_H{house}.csv')

            if not os.path.exists(aggregate_file) or not os.path.exists(appliance_file):
                continue

            # Load data
            aggregate_data = pd.read_csv(aggregate_file)
            appliance_data = pd.read_csv(appliance_file)


            aggregate_data.set_index('time', inplace=True)
            appliance_data.set_index('time', inplace=True)

            merged_data = aggregate_data.join(appliance_data, how='outer')
            merged_data.index = pd.to_datetime(merged_data.index)
            merged_data = merged_data.resample('8S').mean().fillna(method='backfill', limit=1)
            merged_data.dropna(inplace=True)
            merged_data.reset_index(inplace=True)

            house_data_map[house] = merged_data
        return house_data_map

    def saveData(self, dataframe, house_number):
        """
        Save data to appropriate directories for train, validation, or test sets.
        """
        os.makedirs(self.save_path, exist_ok=True)
        output_file = os.path.join(self.save_path, f'{self.appliance_name_formatted}_H{house_number}.csv')
        dataframe.to_csv(output_file, index=False)
        if self.debug:
            print(f"Saved data for House {house_number} to {output_file}")

    def createData(self):
        for house in self.house_data_map:
            data = self.house_data_map[house]
            print(data)
            data = data[:min(self.num_rows, len(data))]
            self.saveData(data, house)



ukdale_appliances = ["microwave", "dishwasher", "fridge", "kettle", "washing machine"]
redd_appliances = ["microwave", "dishwasher", "fridge"]
for appliance in test:
    ukdale_appliance_manager = DatasetManager(
        data_directory=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "UKDALE_data_separated"),
        save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "ukdale_appliance_data"),
        dataset='ukdale',
        appliance_name=appliance,
        debug=True,
        max_num_houses=4,
    )
    ukdale_appliance_manager.createData()

