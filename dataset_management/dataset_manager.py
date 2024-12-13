import pandas as pd
import os
import random

class DatasetManager:
    def __init__(self, data_directory, save_path, appliance_name, debug=False, max_num_houses=None, num_days=10):
        self.data_directory = data_directory
        self.save_path = save_path
        self.appliance_name = appliance_name.lower()
        self.debug = debug
        self.max_num_houses = max_num_houses
        self.sample_seconds = 8
        self.num_rows = int(num_days * 24 * 60 * 60 / self.sample_seconds)

        # Dynamically load house data
        self.house_data_map = self.loadData()
        self.train_house, self.validation_house, self.test_houses = self.splitHouses()



    def loadData(self):
        """
        Load data for all houses and create a mapping of house number to DataFrame.
        """
        appliance_name_formatted = self.appliance_name.replace(" ", "_").lower()
        house_data_map = {}
        num_houses_loaded = 0

        for house_dir in os.listdir(self.data_directory):
            if self.max_num_houses and num_houses_loaded >= self.max_num_houses:
                break

            house_number = house_dir.split("_")[1]  # Extract house number
            house_path = os.path.join(self.data_directory, house_dir)
            aggregate_file = os.path.join(house_path, f'aggregate_H{house_number}.csv')
            appliance_file = os.path.join(house_path, f'{appliance_name_formatted}_H{house_number}.csv')

            if not os.path.exists(aggregate_file) or not os.path.exists(appliance_file):
                continue

            # Load data
            aggregate_data = pd.read_csv(aggregate_file)
            appliance_data = pd.read_csv(appliance_file)


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
            merged_data = merged_data[['time', 'aggregate', appliance_name_formatted]]
            house_data_map[int(house_number)] = merged_data
            num_houses_loaded += 1

        if len(house_data_map) < 2:
            raise ValueError("Not enough houses to create a dataset.")

        return house_data_map

    def normalizeData(self, data):
        """
        Normalize aggregate and appliance data globally for all houses.
        """
        appliance_name_formatted = self.appliance_name.replace(" ", "_").lower()

        data['aggregate'] = (data['aggregate'] - data['aggregate'].min()) / (data['aggregate'].max() - data['aggregate'].min())

            # Normalize appliance data
        data[appliance_name_formatted] = (data[appliance_name_formatted] - data[appliance_name_formatted].min() )/ (data[appliance_name_formatted].max() - data[appliance_name_formatted].min())

        return data

            

    def splitHouses(self):
        """
        Split houses into train, validation, and test sets.
        """
        house_numbers = list(self.house_data_map.keys())
        random.shuffle(house_numbers)

        # Assign one house for validation and one for testing
        validation_house = house_numbers.pop()
        train_house = house_numbers.pop()
        test_houses = house_numbers  # Remaining houses

        if self.debug:
            print(f"Train house: {train_house}")
            print(f"Validation house: {validation_house}")
            print(f"Test houses: {test_houses}")

        return train_house, validation_house, test_houses

    def saveData(self, dataframe, set_type, house_number):
        """
        Save data to appropriate directories for train, validation, or test sets.
        """
        appliance_name_formatted = self.appliance_name.replace(" ", "_").lower()
        os.makedirs(self.save_path, exist_ok=True)
        output_file = os.path.join(self.save_path, f'{appliance_name_formatted}_{set_type}_H{house_number}.csv')
        dataframe['time'] = pd.to_datetime(dataframe['time'])  
        dataframe = dataframe.set_index('time')
        dataframe = dataframe.resample(f'{self.sample_seconds}S').mean().fillna(method='backfill', limit=1)
        dataframe.reset_index(inplace=True)
        dataframe = dataframe[dataframe['aggregate'] >= dataframe[appliance_name_formatted]]
        dataframe = dataframe.head(self.num_rows)
        dataframe = self.normalizeData(dataframe)
        # remove rows where the aggregate is less than the appliance values 
        dataframe.to_csv(output_file, index=False)
        if self.debug:
            print(f"Saved {set_type} data for House {house_number} to {output_file}")

    def createTrainSet(self):
        train_data = self.house_data_map[self.train_house]
        self.saveData(train_data, 'train', self.train_house)

    def createValidationSet(self):
        validation_data = self.house_data_map[self.validation_house]
        self.saveData(validation_data, 'validation', self.validation_house)

    def createTestSet(self):
        for house_number in self.test_houses:
            train_data = self.house_data_map[house_number]
            self.saveData(train_data, 'test', house_number)

refit_data_manager = DatasetManager(
    data_directory=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "REFIT_data_separated"),
    save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "REFIT_data_microwave"),
    appliance_name='microwave',
    debug=True,
    max_num_houses=4
)

refit_data_manager.createTrainSet()
refit_data_manager.createValidationSet()
refit_data_manager.createTestSet()