from abc import ABC, abstractmethod
import pandas as pd
import os
import nilmtk
import appliance_parameters
import random


class DatasetManager(ABC):
    def __init__(self, data_directory, save_path, sample_seconds = 8, debug = False, appliance_name = 'microwave', max_num_houses = 5):
        self.data_directory = data_directory
        self.save_path = save_path
        self.sample_seconds = sample_seconds
        self.debug = debug
        self.appliance_name = appliance_name.lower()
        self.AGGREGATE_COLUMN = 'aggregate'
        self.APPLIANCE_COLUMN = self.appliance_name
        self.TIME_COLUMN = 'time'
        self.max_num_houses = max_num_houses
        self.house_data_map = self.loadData()
        self.normalizeData()
        self.train_houses, self.validation_house, self.test_house = self.splitHouses(self.house_data_map)
        # sets the maximum number of houses to fetch data for

    def getApplianceMeanAndStd(self):
        """
        Calculate the global mean and standard deviation for the appliance across all houses
        """
        # Concatenate all appliance data across houses into one Series
        all_appliance_data = pd.concat([house[self.APPLIANCE_COLUMN] for house in self.house_data_map.values()])

        # Compute mean and std directly
        global_mean = all_appliance_data.mean()
        global_std = all_appliance_data.std()

        return global_mean, global_std


    
    def splitHouses(self, house_data_map, train_ratio=0.7, validation_ratio=0.15, random_seed=None):
        """
        Split houses into one test house, one validation house, and the rest as train houses.
        """
        house_numbers = list(house_data_map.keys())

        # Shuffle for randomness
        if random_seed:
            random.seed(random_seed)
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

    def normalizeData(self):
        appliance_mean = self.getApplianceMeanAndStd()[0]
        appliance_std = self.getApplianceMeanAndStd()[1]
        for house_number, house_data in self.house_data_map.items():
            house_data[self.APPLIANCE_COLUMN] = (house_data[self.APPLIANCE_COLUMN] - appliance_mean) / appliance_std
            house_data[self.AGGREGATE_COLUMN] = (house_data[self.AGGREGATE_COLUMN] - house_data[self.AGGREGATE_COLUMN].mean()) / house_data[self.AGGREGATE_COLUMN].std()
            

    @abstractmethod
    def loadData(self):
        # Load data from the data directory specified
        raise NotImplementedError

    @abstractmethod
    def createTrainSet(self):
        # Create a training set
        raise NotImplementedError

    @abstractmethod
    def createValidationSet(self):
        # Create a validation set
        raise NotImplementedError

    @abstractmethod
    def createTestSet(self):
        # Create a test set
        raise NotImplementedError

    @abstractmethod
    def saveData(self):
        raise NotImplementedError



class REFITDataManager(DatasetManager):
    def __init__(self, data_directory, save_path, sample_seconds = 8, debug = False, appliance_name = 'microwave', max_num_houses = 5):
        super().__init__(data_directory, save_path, sample_seconds, debug, appliance_name, max_num_houses)
        if self.debug:
            print(f'Using appliance {self.appliance_name}')

    def loadData(self):
        house_data_map = {}
        num_houses_loaded = 0
        for house in os.listdir(self.data_directory):
            house_path = os.path.join(self.data_directory, house)

            house_number = house.split('_')[1]

            aggregate_file = os.path.join(house_path, f'aggregate_H{house_number}.csv')
            appliance_file = os.path.join(house_path, f'{self.appliance_name}_H{house_number}.csv')

            if not os.path.exists(aggregate_file) or not os.path.exists(appliance_file):
                continue

            aggregate_data = pd.read_csv(aggregate_file)
            appliance_data = pd.read_csv(appliance_file)
            aggregate_data['time'] = pd.to_datetime(aggregate_data['time'])
            appliance_data['time'] = pd.to_datetime(appliance_data['time'])


            print(aggregate_data['time'].dtype)
            print(appliance_data['time'].dtype)

            merged_data = pd.merge_asof(aggregate_data.sort_values(by='time'), appliance_data.sort_values(by='time'), on='time', direction='nearest')

            merged_data['time'] = pd.to_datetime(merged_data['time'])

            merged_data = merged_data[[self.TIME_COLUMN, self.AGGREGATE_COLUMN, self.APPLIANCE_COLUMN]]

            house_data_map[int(house_number)] = merged_data
            num_houses_loaded += 1
            if num_houses_loaded >= self.max_num_houses:
                break

        if len(house_data_map) < 2:
            raise ValueError('Not enough houses to create a dataset')    
        return house_data_map     

    def saveData(self, dataframe, type, house_number):
        """
        Takes a dataframe and the type of data (0 (train), 1 (validation), 2 (test) ) as input and saves it to the save_path
        """
        os.makedirs(self.save_path, exist_ok = True)
        data_type_mapping = {0: 'train', 1: 'validation', 2: 'test'}
        data_type = data_type_mapping[type]
        if self.debug:
            print(f'Saving {data_type} data for house {house_number}')
        dataframe.to_csv(os.path.join(self.save_path,f'{self.appliance_name}_{data_type}_H{str(house_number)}.csv'), index=False)
    
    def createTrainSet(self):
        for house_number in self.train_houses:
            train_data = self.house_data_map[house_number]
            self.saveData(train_data, 0, house_number)

    def createValidationSet(self):
        validation_data = self.house_data_map[self.validation_house]
        self.saveData(validation_data, 1, self.validation_house)
            
    def createTestSet(self):
        test_data = self.house_data_map[self.test_house]
        self.saveData(test_data, 2, self.test_house)



class UKDALEDataManager(DatasetManager):
    def __init__(self, data_directory, save_path, sample_seconds=8, debug=False, appliance_name='microwave', validation_percent=13):
        super().__init__(data_directory, save_path, sample_seconds, debug, appliance_name)
        self.validation_percent = validation_percent
        self.params_appliance = appliance_parameters.ukdale_params_appliance
        if self.appliance_name not in self.params_appliance:
            self.appliance_name = 'microwave'
        if self.debug:
            print(f'Using appliance {self.appliance_name}')

    def loadData(self, house_number, channel_number):
        """
        loads UKDALE data for the house number and channel number (appliance) specified
        """
        if self.debug:
            print(f'Loading data for house {house_number} and channel {channel_number}')
        dataset = nilmtk.DataSet(self.data_directory)
        mains_data = dataset.buildings[house_number].elec.mains().power_series_all_data()
        appliance_data = dataset.buildings[house_number].elec[channel_number].power_series_all_data()
        mains_df = pd.DataFrame({'time': mains_data.index, 'aggregate': mains_data.values})
        appliance_df = pd.DataFrame({'time': appliance_data.index, self.appliance_name: appliance_data.values})
        mains_df = mains_df.sort_values(by='time')
        appliance_df = appliance_df.sort_values(by='time')
        merged_df = pd.merge_asof(mains_df, appliance_df, on='time', direction='nearest')
        merged_df['time'] = pd.to_datetime(merged_df['time'])
        return merged_df

    def saveData(self, dataframe, type):
        os.makedirs(self.save_path, exist_ok = True)
        data_type_mapping = {0: 'train', 1: 'validation', 2: 'test'}
        data_type = data_type_mapping[type]
        if self.debug:
            print(f'Saving {data_type} data for appliance {self.appliance_name}')
        dataframe.to_csv(os.path.join(self.save_path,f'{self.appliance_name}_{data_type}.csv'), index=False, header=False)

    def createTrainSet(self):
        train_house = self.params_appliance[self.appliance_name]['train_house']
        train_house_index = self.params_appliance[self.appliance_name]['houses'].index(train_house)
        appliance_channel = self.params_appliance[self.appliance_name]['channels'][train_house_index]
        train_data = self.loadData(train_house, appliance_channel)
        train_data = self.normalizeData(train_data)

        validation_index = (self.validation_percent * len(train_data)) // 100
        self.createValidationSet(train_data, validation_index)
        train_data = train_data[:validation_index]
        self.saveData(train_data, 0)


    def createValidationSet(self, train_data, validation_index):
        validation_data = train_data[validation_index:]
        self.saveData(validation_data, 1)

    def createTestSet(self):
        test_house = self.params_appliance[self.appliance_name]['test_house']
        test_house_index = self.params_appliance[self.appliance_name]['houses'].index(test_house)
        appliance_channel = self.params_appliance[self.appliance_name]['channels'][test_house_index]
        test_data = self.loadData(test_house, appliance_channel)
        test_data = self.normalizeData(test_data)
        self.saveData(test_data, 2)

class REDDDataManager(DatasetManager):
    def __init__(self, data_directory, save_path, sample_seconds=8, debug=False, appliance_name='microwave'):
        super().__init__(data_directory, save_path, sample_seconds, debug, appliance_name)
        self.params_appliance = appliance_parameters.redd_params_appliance

        if self.appliance_name not in self.params_appliance:
            self.appliance_name = 'microwave'

        if self.debug:
            print(f'Using appliance {self.appliance_name}')
    
    def loadData(self, house_number, channel_number):
        pass

    def saveData(self, dataframe, type):
        pass

    def createTrainSet(self):
        pass

    def createValidationSet(self):
        pass

    def createTestSet(self):
        pass
    

# UKDALEManager = UKDALEDataManager(os.path.join('C:\\', 'Users', 'yashm', 'Downloads', 'UKDALE', 'ukdale.h5'), 'ukdaledata/', debug = True, appliance_name='microwave')
refitManager = REFITDataManager(os.path.join('C:\\', 'Users', 'yashm', 'Repos Personal', 'seq2point-update', 'updated_dataset_management', 'refit_data_separated'), os.path.join('C:\\', 'Users', 'yashm', 'Repos Personal', 'seq2point-update', 'updated_dataset_management', 'refit_house_data'), debug=True, appliance_name='computer', max_num_houses=4)
print(refitManager.house_data_map)
refitManager.createTrainSet()
refitManager.createValidationSet()
refitManager.createTestSet()