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
        self.num_rows = 10**5 
        self.house_data_map = self.loadData()
        self.train_houses, self.validation_house, self.test_houses = self.splitHouses()



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

            # Remove rows where the appliance value is -1 (ECO data placeholder for missing data)
            merged_data = merged_data[merged_data[appliance_name_formatted] != -1]
            
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
        
        # calculate the global min and max of the aggregate data from training houses
        aggregate_min = min([self.house_data_map[house_number]['aggregate'].min() for house_number in self.train_houses])
        aggregate_max = max([self.house_data_map[house_number]['aggregate'].max() for house_number in self.train_houses])

        data['aggregate'] = (data['aggregate'] - aggregate_min) / (aggregate_max - aggregate_min)

         # Normalize appliance data
        data[appliance_name_formatted] = (data[appliance_name_formatted] - aggregate_min )/ (aggregate_max - aggregate_min)

        return data

            

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
        appliance_name_formatted = self.appliance_name.replace(" ", "_").lower()
        os.makedirs(self.save_path, exist_ok=True)
        output_file = os.path.join(self.save_path, f'{appliance_name_formatted}_{set_type}_H{house_number}.csv')
        # remove rows where the aggregate is less than the appliance values 
        dataframe = dataframe[dataframe['aggregate'] >= dataframe[appliance_name_formatted]]
        dataframe = self.normalizeData(dataframe)

        dataframe.to_csv(output_file, index=False)
        if self.debug:
            print(f"Saved {set_type} data for House {house_number} to {output_file}")

    def createTrainSet(self):
        for house_number in self.train_houses:
            # Find the window with the most appliance activity to avoid sparse data
            appliance_name_formatted = self.appliance_name.replace(" ", "_").lower()
            train_data = self.house_data_map[house_number]
            window_size = self.num_rows  
            step_size = window_size // 2
            maximum_activity = 0 
            best_window_start = None
            best_window_end = None
            train_data['abs_change'] = train_data[appliance_name_formatted].diff().abs()

   
            for start in range(0, len(train_data) - window_size + 1, step_size):
                end = start + window_size

                # Extract the current window
                window = train_data.iloc[start:end]
                activity = window['abs_change'].sum()

                # Check if this window has higher activity
                if activity > maximum_activity:
                    if self.debug:
                        print(f"New max activity: {activity} at window {start}-{end}")
                    maximum_activity = activity
                    best_window_start = start
                    best_window_end = end

            # Use the best window found
            if best_window_start is not None and best_window_end is not None:
                train_data = train_data.iloc[best_window_start:best_window_end]
            else:
                train_data = train_data[:window_size]

            # Drop temporary column and save the data
            train_data.drop(columns=['abs_change'], inplace=True)
            self.saveData(train_data, 'train', house_number)

    def createValidationSet(self):
        validation_data = self.house_data_map[self.validation_house]
        validation_data = validation_data[:min(self.num_rows, len(validation_data))]
        self.saveData(validation_data, 'validation', self.validation_house)

    def createTestSet(self):
        for house_number in self.test_houses:
            test_data = self.house_data_map[house_number]
            test_data = test_data[:min(self.num_rows, len(test_data))]
            self.saveData(test_data, 'test', house_number)

ukdale_data_manager = DatasetManager(
    data_directory=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "UKDALE_data_separated"),
    save_path=os.path.join("C:\\", "Users", "yashm", "OneDrive - The University of Manchester", "Documents", "UKDALE_data_kettle"),
    appliance_name='kettle',
    debug=True,
)

ukdale_data_manager.createTrainSet()
ukdale_data_manager.createValidationSet()
ukdale_data_manager.createTestSet()