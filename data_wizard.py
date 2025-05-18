from dataset_management.data_separation.data_separator import DataSeparator
from dataset_management.dataset_manager import DatasetManager
from dataset_management.dataset_registry import DatasetRegistry
import sys
import os
import json




def runDataSeparator():
    print("Data Separator")

    file_path = input("Enter the file path to the raw data: ")

    save_path = input("Enter the save path of the separated data: ")

    num_houses = input("Enter the maximum number of houses to process (press enter to process all): ")
    try:
        num_houses = int(num_houses)
    except ValueError:
        num_houses = None
    
    dataset_type = input("Enter the dataset type: ")
    if dataset_type.lower() not in DatasetRegistry.getDatasets():
        print("Invalid dataset type. Please try again.")
        return
    
    appliance = input("Enter the appliance to filter for (press enter to process all): ")
    if appliance not in DatasetRegistry.getAvailableAppliances(dataset_type):
        appliance = None

    data_separator = DataSeparator(file_path=file_path, save_path=save_path, dataset_type=dataset_type, appliance_name=appliance, num_houses=num_houses)
    data_separator.process_data()
        
def runDatasetManager():
    data_directory = input("Enter the path of the separated data: ")
    save_path = input("Enter the save path: ")
    dataset_type = input("Enter the dataset type: ")

    if dataset_type.lower() not in DatasetRegistry.getDatasets():
        print("Invalid dataset type. Please try again.")
        return
    appliance = input("Enter the appliance name: ")

    if appliance not in DatasetRegistry.getAvailableAppliances(dataset_type):
        print("Invalid appliance name. Please try again.")
        return
    
    debug = input("Enter debug mode (y/n): ")
    if debug.lower() == 'y':
        debug = True
    else:
        debug = False
    
    max_num_houses = input("Enter the maximum number of houses to process (press enter to process all): ")
    try:
        max_num_houses = int(max_num_houses)
    except ValueError:
        max_num_houses = None
    
    max_num_rows = input("Enter the maximum number of rows to process (press enter to process all): ")
    try:
        max_num_rows = int(max_num_rows)
    except ValueError:
        max_num_rows = 1 * (10**6)
    
    dataset_manager = DatasetManager(data_directory=data_directory, save_path=save_path, dataset=dataset_type, appliance_name=appliance, debug=debug, max_num_houses=max_num_houses, max_num_rows=max_num_rows)
    dataset_manager.createData()


def main():
    while True:
        print("Select an option:")
        print("1. Run Data Separator")
        print("2. Run Dataset Manager")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            runDataSeparator()
        elif choice == '2':
            runDatasetManager()
        elif choice == '3':
            print("Exiting...")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()