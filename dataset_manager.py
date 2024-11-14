from abc import ABC, abstractmethod
class DatasetManager(ABC):
    def __init__(self, data_directory, appliance_name, save_path, file_extension, sample_seconds = 8, validation_percent = 10, debug = False):
        self.data_directory = data_directory
        self.appliance_name = appliance_name
        self.save_path = save_path
        self.file_extension = file_extension
        self.sample_seconds = sample_seconds
        self.validation_percent = validation_percent
        self.params_appliance = {}
        self.debug = debug

    @abstractmethod
    def loadData(self):
        # Load data from the data directory specified
        raise NotImplementedError
    
    @abstractmethod
    def createTrainset(self):
        # Create a training set
        raise NotImplementedError

    @abstractmethod
    def createTestset(self):
        # Create a test set
        raise NotImplementedError

    def saveData(self):
        # Save data to the save path as a CSV file
        pass

    def normalizeData(self):
        # Normalize the data
        pass