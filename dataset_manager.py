from abc import ABC, abstractmethod
import pandas as pd
import os
import nilmtk
class DatasetManager(ABC):
    def __init__(self, data_directory, save_path, sample_seconds = 8, debug = False, appliance_name = 'microwave'):
        self.data_directory = data_directory
        self.save_path = save_path
        self.sample_seconds = sample_seconds
        self.params_appliance = {}
        self.debug = debug
        self.appliance_name = appliance_name.lower()
        self.AGGREGATE_COLUMN = 'aggregate'
        self.APPLIANCE_COLUMN = self.appliance_name
        self.TIME_COLUMN = 'time'

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

    def normalizeData(self, dataframe):
        """
        Takes a dataframe as input with columns 'aggregate' 
        and 'appliance_name' and normalizes the data
        returns a dataframe with normalized data
        """
        appliance_mean = self.params_appliance[self.appliance_name]['mean']
        appliance_std = self.params_appliance[self.appliance_name]['std']
        dataframe['aggregate'] = (dataframe['aggregate'] - dataframe['aggregate'].mean()) / (dataframe['aggregate'].std())
        dataframe[self.appliance_name] = (dataframe[self.appliance_name] - appliance_mean) / appliance_std
        return dataframe


class REFITDataManager(DatasetManager):
    def __init__(self, data_directory, save_path, sample_seconds = 8, debug = False, appliance_name = 'microwave'):
        super().__init__(data_directory, save_path, sample_seconds, debug, appliance_name)
        self.params_appliance = {
            'kettle': {
                'windowlength': 599,
                'on_power_threshold': 2000,
                'max_on_power': 3998,
                'mean': 700,
                'std': 1000,
                's2s_length': 128,
                'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
                'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
                'test_house': 2,
                'validation_house': 5,
                'test_on_train_house': 5,
            },
            'microwave': {
                'windowlength': 599,
                'on_power_threshold': 200,
                'max_on_power': 3969,
                'mean': 500,
                'std': 800,
                's2s_length': 128,
                'houses': [4, 10, 12, 17, 19],
                'channels': [8, 8, 3, 7, 4],
                'test_house': 4,
                'validation_house': 17,
                'test_on_train_house': 10,
            },
            'fridge': {
                'windowlength': 599,
                'on_power_threshold': 50,
                'max_on_power': 3323,
                'mean': 200,
                'std': 400,
                's2s_length': 512,
                'houses': [2, 5, 9, 12, 15],
                'channels': [1, 1, 1,  1, 1],
                'test_house': 15,
                'validation_house': 12,
                'test_on_train_house': 5,
            },
            'dishwasher': {
                'windowlength': 599,
                'on_power_threshold': 10,
                'max_on_power': 3964,
                'mean': 700,
                'std': 1000,
                's2s_length': 1536,
                'houses': [5, 7, 9, 13, 16, 18, 20],
                'channels': [4, 6, 4, 4, 6, 6, 5],
                'test_house': 20,
                'validation_house': 18,
                'test_on_train_house': 13,
            },
            'washingmachine': {
                'windowlength': 599,
                'on_power_threshold': 20,
                'max_on_power': 3999,
                'mean': 400,
                'std': 700,
                's2s_length': 2000,
                'houses': [2, 5, 7, 8, 9, 15, 16, 17, 18],
                'channels': [2, 3, 5, 4, 3, 3, 5, 4, 5],
                'test_house': 8,
                'validation_house': 18,
                'test_on_train_house': 5,
            }
            }
        if self.appliance_name not in self.params_appliance:
            self.appliance_name = 'microwave'
        if self.debug:
            print(f'Using appliance {self.appliance_name}')

    def loadData(self, house_number, channel_number):
        """
        loads REFIT data for the house number and channel number (appliance) specified
        """
        if self.debug:
            print(f'Loading data for house {house_number} and channel {channel_number}')
        file_name = os.path.join(self.data_directory, f'CLEAN_House{house_number}.csv')
        data = pd.read_csv(file_name,
                           header = 0,
                           names = [self.TIME_COLUMN, self.AGGREGATE_COLUMN, self.APPLIANCE_COLUMN],
                           usecols = [0,2, channel_number + 2],
                           na_filter = False,
                           parse_dates = True,
                           memory_map = True)
        data['time'] = pd.to_datetime(data['time'])
        return data

    def saveData(self, dataframe, type, house_number):
        """
        Takes a dataframe and the type of data (0 (train), 1 (validation), 2 (test) ) as input and saves it to the save_path
        """
        os.makedirs(self.save_path, exist_ok = True)
        data_type_mapping = {0: 'train', 1: 'validation', 2: 'test'}
        data_type = data_type_mapping[type]
        if self.debug:
            print(f'Saving {data_type} data for house {house_number}')
        dataframe.to_csv(os.path.join(self.save_path,f'{self.appliance_name}_{data_type}_H{str(house_number)}.csv'), index=False, header=False)
    
    def createTrainSet(self):
        train_houses = []
        for house in self.params_appliance[self.appliance_name]['houses']:
            if house != self.params_appliance[self.appliance_name]['test_house'] and house != self.params_appliance[self.appliance_name]['validation_house']:
                train_houses.append(house)
        # WHY IS THE ORIGINAL SCRIPT USING TEST_ON_TRAIN INSTEAD OF JUST THE TRAIN HOUSES?  
        for train_house in train_houses:
            if self.debug:
                print(f'Creating training set for house {train_house}')
            appliance_channel = self.params_appliance[self.appliance_name]['channels'][self.params_appliance[self.appliance_name]['houses'].index(train_house)]
            data = self.loadData(train_house, appliance_channel)
            normalized_data = self.normalizeData(data)
            self.saveData(normalized_data, 0, train_house)

    def createValidationSet(self):
            
        validation_house = self.params_appliance[self.appliance_name]['validation_house']
        if self.debug:
            print(f'Creating validation set for house {validation_house}')
        appliance_channel = self.params_appliance[self.appliance_name]['channels'][self.params_appliance[self.appliance_name]['houses'].index(validation_house)]
        data = self.loadData(validation_house, appliance_channel)
        normalized_data = self.normalizeData(data)
        self.saveData(normalized_data, 1, validation_house)
            
    def createTestSet(self):
        test_house = self.params_appliance[self.appliance_name]['test_house']
        if self.debug:
            print(f'Creating test set for house {test_house}')
        appliance_channel = self.params_appliance[self.appliance_name]['channels'][self.params_appliance[self.appliance_name]['houses'].index(test_house)]
        data = self.loadData(test_house, appliance_channel)
        normalized_data = self.normalizeData(data)
        self.saveData(normalized_data, 2, test_house)


class UKDALE(DatasetManager):
    def __init__(self, data_directory, save_path, sample_seconds=8, debug=False, appliance_name='microwave', validation_percent=13):
        super().__init__(data_directory, save_path, sample_seconds, debug, appliance_name)
        self.validation_percent = validation_percent
        self.params_appliance = {
        'kettle': {
            'windowlength': 599,
            'on_power_threshold': 2000,
            'max_on_power': 3998,
            'mean': 700,
            'std': 1000,
            's2s_length': 128,
            'houses': [1, 2],
            'channels': [10, 8],
            'train_house': 1,
            'test_house': 2,
        },
        'microwave': {
            'windowlength': 599,
            'on_power_threshold': 200,
            'max_on_power': 3969,
            'mean': 500,
            'std': 800,
            's2s_length': 128,
            'houses': [1, 2],
            'channels': [13, 15],
            'train_house': 1,
            'test_house': 2,
        },
        'fridge': {
            'windowlength': 599,
            'on_power_threshold': 50,
            'max_on_power': 3323,
            'mean': 200,
            'std': 400,
            's2s_length': 512,
            'houses': [1, 2],
            'channels': [12, 14],
            'train_house': 1,
            'test_house': 2,
        },
        'dishwasher': {
            'windowlength': 599,
            'on_power_threshold': 10,
            'max_on_power': 3964,
            'mean': 700,
            'std': 1000,
            's2s_length': 1536,
            'houses': [1, 2],
            'channels': [6, 13],
            'train_house': 1,
            'test_house': 2,
        },
        'washingmachine': {
            'windowlength': 599,
            'on_power_threshold': 20,
            'max_on_power': 3999,
            'mean': 400,
            'std': 700,
            's2s_length': 2000,
            'houses': [1, 2],
            'channels': [5, 12],
            'train_house': 1,
            'test_house': 2,
        }
        }
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





UKDALEManager = UKDALE(os.path.join('C:\\', 'Users', 'yashm', 'Downloads', 'UKDALE', 'ukdale.h5'), 'ukdaledata/', debug = True, appliance_name='kettle')

UKDALEManager.createTrainSet()
UKDALEManager.createTestSet()

