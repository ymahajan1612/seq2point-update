import pandas as pd
import os
import random
import json

class DatasetManager:
    def __init__(self, data_directory, save_path, dataset, appliance_name, debug=False, max_num_houses=None):
        self.debug = debug
        self.data_directory = data_directory
        self.save_path = save_path
        self.dataset = dataset.lower()
        self.appliance_name = appliance_name.lower()
        self.appliance_name_formatted = self.appliance_name.replace(" ", "_") 

        with open(os.path.join("dataset_management", f"{self.dataset}_parameters.json"), "r") as f:
            parameters = json.load(f)

        self.train_house = parameters[self.appliance_name_formatted]['train_house']
        self.test_house = parameters[self.appliance_name_formatted]['test_house']
        self.houses = parameters[self.appliance_name_formatted]['houses']

        self.num_rows = 1 * (10**6)

        self.house_data_map = self.loadData()

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
            merged_data = merged_data.resample('5S').mean().fillna(method='backfill', limit=1)
            merged_data.dropna(inplace=True)
            merged_data.reset_index(inplace=True)

            house_data_map[house] = merged_data
        return house_data_map
           
    def splitHouses(self):
        """
        Split houses into train, validation, and test sets.
        """
        house_numbers = list(self.house_data_map.keys())
        random.shuffle(house_numbers)

        # Assign one house for validation
        validation_house = house_numbers.pop()

        # Split remaining houses into train and test sets
        num_test_houses = len(house_numbers) // 2
        test_houses = house_numbers[:num_test_houses]
        train_houses = house_numbers[num_test_houses:]

        if self.debug:
            print(f"Train houses: {train_houses}")
            print(f"Validation house: {validation_house}")
            print(f"Test houses: {test_houses}")

        return train_houses, validation_house, test_houses

    def saveData(self, dataframe, set_type, house_number):
        """
        Save data to appropriate directories for train, validation, or test sets.
        """
        os.makedirs(self.save_path, exist_ok=True)
        output_file = os.path.join(self.save_path, f'{self.appliance_name_formatted}_{set_type}_H{house_number}.csv')
        dataframe.to_csv(output_file, index=False)
        if self.debug:
            print(f"Saved {set_type} data for House {house_number} to {output_file}")

    def createTrainSet(self):
        train_data = self.house_data_map[self.train_house]
        train_data = train_data[:min(self.num_rows, len(train_data))]
        self.saveData(train_data, 'train', self.train_house)
        
    def createTestSet(self):
        test_data = self.house_data_map[self.test_house]
        test_data = test_data[:min(self.num_rows, len(test_data))]
        self.saveData(test_data, 'test', self.test_house)

# ukdale_data_manager = DatasetManager(
#     data_directory=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "UKDALE_data_separated"),
#     save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "UKDALE_data_kettle"),
#     dataset='ukdale',
#     appliance_name='kettle',
#     debug=True,
# )

ukdale_appliance = ["microwave", "dishwasher","fridge"]

for appliance in ukdale_appliance:
    print("Appliance: ", appliance)
    ukdale_appliance_manager = DatasetManager(
        data_directory=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "REDD_data_separated"),
        save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "REDD_appliance_data"),
        dataset='REDD',
        appliance_name=appliance,
        debug=True,
    )
    ukdale_appliance_manager.createTrainSet()
    ukdale_appliance_manager.createTestSet()
