import pandas as pd
import os
import random

class DatasetManager:
    def __init__(self, data_directory, save_path, appliance_name, debug=False, max_num_houses=None):
        self.data_directory = data_directory
        self.save_path = save_path
        self.appliance_name = appliance_name.lower()
        self.debug = debug
        self.max_num_houses = max_num_houses

        # Dynamically load house data
        self.house_data_map = self.loadData()
        self.normalizeData()
        self.train_houses, self.validation_house, self.test_house = self.splitHouses()

    def loadData(self):
        """
        Load data for all houses and create a mapping of house number to DataFrame.
        """
        house_data_map = {}
        num_houses_loaded = 0

        for house_dir in os.listdir(self.data_directory):
            if self.max_num_houses and num_houses_loaded >= self.max_num_houses:
                break

            house_number = house_dir.split("_")[1]  # Extract house number
            house_path = os.path.join(self.data_directory, house_dir)
            aggregate_file = os.path.join(house_path, f'aggregate_H{house_number}.csv')
            appliance_file = os.path.join(house_path, f'{self.appliance_name}_H{house_number}.csv')

            if not os.path.exists(aggregate_file) or not os.path.exists(appliance_file):
                continue

            # Load data
            aggregate_data = pd.read_csv(aggregate_file)
            appliance_data = pd.read_csv(appliance_file)

            print(len(aggregate_data))

            # Ensure time columns are datetime
            aggregate_data['time'] = pd.to_datetime(aggregate_data['time'])
            appliance_data['time'] = pd.to_datetime(appliance_data['time'])

            # Merge on time
            merged_data = pd.merge_asof(
                aggregate_data.sort_values(by='time'),
                appliance_data.sort_values(by='time'),
                on='time',
                direction='nearest'
            )
            merged_data = merged_data[['time', 'aggregate', self.appliance_name]]

            house_data_map[int(house_number)] = merged_data
            num_houses_loaded += 1

        if len(house_data_map) < 2:
            raise ValueError("Not enough houses to create a dataset.")

        return house_data_map

    def normalizeData(self):
        """
        Normalize aggregate and appliance data globally for all houses.
        """
        all_appliance_data = pd.concat([df[self.appliance_name] for df in self.house_data_map.values()])
        appliance_mean = all_appliance_data.mean()
        appliance_std = all_appliance_data.std()

        for house_data in self.house_data_map.values():
            house_data['aggregate'] = (house_data['aggregate'] - house_data['aggregate'].mean()) / house_data['aggregate'].std()
            house_data[self.appliance_name] = (house_data[self.appliance_name] - appliance_mean) / appliance_std

    def splitHouses(self):
        """
        Split houses into train, validation, and test sets.
        """
        house_numbers = list(self.house_data_map.keys())
        random.shuffle(house_numbers)

        # Assign one house for validation and one for testing
        validation_house = house_numbers.pop()
        test_house = house_numbers.pop()
        train_houses = house_numbers  # Remaining houses

        if self.debug:
            print(f"Train houses: {train_houses}")
            print(f"Validation house: {validation_house}")
            print(f"Test house: {test_house}")

        return train_houses, validation_house, test_house

    def saveData(self, dataframe, set_type, house_number):
        """
        Save data to appropriate directories for train, validation, or test sets.
        """
        set_dir = os.path.join(self.save_path, set_type)
        os.makedirs(set_dir, exist_ok=True)

        output_file = os.path.join(set_dir, f'{self.appliance_name}_H{house_number}.csv')
        dataframe.to_csv(output_file, index=False)
        if self.debug:
            print(f"Saved {set_type} data for House {house_number} to {output_file}")

    def createTrainSet(self):
        for house_number in self.train_houses:
            train_data = self.house_data_map[house_number]
            self.saveData(train_data, 'train', house_number)

    def createValidationSet(self):
        validation_data = self.house_data_map[self.validation_house]
        self.saveData(validation_data, 'validation', self.validation_house)

    def createTestSet(self):
        test_data = self.house_data_map[self.test_house]
        self.saveData(test_data, 'test', self.test_house)


# refitManager = DatasetManager(os.path.join("C:\\", "Users", "yashm", "Downloads", "refit_data_separated"), os.path.join("C:\\", "Users", "yashm", "Downloads", "refit_data"),"kettle", debug=True, max_num_houses=4)


ukDaleManager = DatasetManager(os.path.join("C:\\", "Users", "yashm", "Downloads", "ukdale_data_separated"), os.path.join("C:\\", "Users", "yashm", "Downloads", "ukdale_data"),"microwave", debug=True)
ukDaleManager.createTrainSet()
ukDaleManager.createValidationSet()
ukDaleManager.createTestSet()