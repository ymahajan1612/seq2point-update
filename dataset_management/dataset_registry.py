import os
import json
class DatasetRegistry:
    """
    Centralized list of available datasets.
    """

    datasets = ["UKDALE", "REDD", "REFIT", "ECO"]

    @classmethod
    def getDatasets(cls):
        """ Returns the list of available datasets. """
        return cls.datasets
    
    @classmethod
    def getAvailableAppliances(cls, dataset):
        """ Returns the list of available appliances for the specified dataset. """
        appliance_mappings_dir = os.path.join("dataset_management","data_separation", f"{dataset}_appliance_mappings.json")
        appliances = set()
        with open(appliance_mappings_dir, 'r') as f:
            appliance_mappings = json.load(f)
        for house in appliance_mappings:
            appliance_list = appliance_mappings[house].values()
            for appliance in appliance_list:
                appliances.add(appliance.lower())
        return appliances
