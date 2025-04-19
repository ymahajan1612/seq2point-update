from model_pipeline import seq2Point_factory, train_model, test_model, finetune_model
import os
import sys

def runningColab():
    """
    Check if the code is running in Google Colab.
    :return: Boolean indicating if the code is running in Google Colab
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def selectCSVFiles(type, using_colab, single_file=False):
    """
    Selects files for training or testing based on the type specified.
    :param type: Type of operation ('train','val','test')
    :param running_in_colab: Boolean indicating if the code is running in Google Colab
    :return: List of selected file paths
    """
    print(f"Please upload the {type} CSV files.")
    if using_colab:
        from google.colab import files
        uploaded = files.upload()
        csv_files = [f"/content/{fname}" for fname in uploaded.keys() if fname.endswith(".csv")]

        if not csv_files:
            print("No CSV files were uploaded. Please try again.")
        
        if single_file and len(csv_files) > 1:
            print("Multiple files were uploaded, but only one is needed. Using the first one.")
            csv_files = [csv_files[0]]
        return csv_files
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if single_file:
            file_path = filedialog.askopenfilename(title=f"Select {type} CSV file", filetypes=[("CSV files", "*.csv")])
            if not file_path:
                print(f"No {type} CSV file was selected. Please try again.")
                return []
            return [file_path]
        else:
            file_paths = filedialog.askopenfilenames(title=f"Select {type} CSV files", filetypes=[("CSV files", "*.csv")])
            if not file_paths:
                print(f"No {type} CSV files were selected. Please try again.")
            return file_paths
    
def selectModelFile(using_colab):
    """
    Selects the model file for evaluation.
    :param using_colab: Boolean indicating if the code is running in Google Colab
    :return: Path to the selected model file
    """
    print("Please upload the model file.")
    if using_colab:
        from google.colab import files
        uploaded = files.upload()
        model_file_path = list(uploaded.keys())[0]
        return f"/content/{model_file_path}"
    else:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        model_file_path = filedialog.askopenfilename(title="Select model file", filetypes=[("Model files", "*.pth")])
        if not model_file_path:
            print("No model file was selected. Please try again.")
            return None
        return model_file_path
    
def trainModelCLI():
    """
    Command Line Interface for training the Seq2Point model.
    """
    print("Training a Seq2Point model...")
    using_colab = runningColab()
    train_csv_dirs = selectCSVFiles('train', using_colab)
    validation_csv_dirs = selectCSVFiles('val', using_colab)

    if not train_csv_dirs or not validation_csv_dirs:
        print("No files selected. Exiting.")
        return

    appliance = input("Enter the appliance name: ")
    dataset = input("Enter the dataset name: ")
    
    available_models = seq2Point_factory.Seq2PointFactory.getModelMappings()
    numbered_models = {str(i+1): model for i, model in enumerate(available_models.keys())}
    if using_colab:
        model_save_dir = "/content/"
    else:
        model_save_dir = os.path.join(os.getcwd(), "saved_models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
    print("Available models:")
    for num, model in numbered_models.items():
        print(f"{num}: {model}")
    model_num = input("Select a model by number: ")
    input_window_length = int(input("Enter the input window length (default is 599): ") or 599)
    num_epochs = int(input("Enter the number of epochs (default is 10): ") or 10)
    model_trainer = train_model.Trainer(model_name=numbered_models[model_num], 
                                        train_csv_dirs = train_csv_dirs,
                                        validation_csv_dirs = validation_csv_dirs,
                                        appliance=appliance,
                                        dataset=dataset,
                                        model_save_dir=model_save_dir,
                                        window_length=input_window_length
                                        )
    model_trainer.trainModel(num_epochs)
    model_trainer.plotLosses()
    print("Model training completed.")
    
    
def evaluateModelCLI():
    """
    Command Line Interface for evaluating the Seq2Point model.
    """
    print("Evaluating a Seq2Point model...")
    using_colab = runningColab()
    test_csv_dir = selectCSVFiles('test', using_colab, single_file=True)[0]
    if not test_csv_dir:
        print("No test file selected. Exiting.")
        return
    model_file_path = selectModelFile(using_colab)
    
    model_tester = test_model.Tester(model_state_dir=model_file_path, test_csv_dir = test_csv_dir)
    model_tester.testModel()
    model_tester.plotResults()
    print("Model evaluation completed.")
    print("MAE: ", model_tester.getMetrics()[0])
    print("SAE: ", model_tester.getMetrics()[1])
    print("Inference Time: ", model_tester.getMetrics()[2])
    print("Model testing completed.")
    results = model_tester.getResults()
    results.to_csv(f"{model_tester.appliance_name_formatted}_results.csv", index=False)

    print(f"Results saved to {model_tester.appliance_name_formatted}_results.csv")

def fineTuneModelCLI():
    using_colab = runningColab()
    finetune_csv_dir = selectCSVFiles('finetune', using_colab, single_file=True)[0]
    if not finetune_csv_dir:
        print("No finetune file selected. Exiting.")
        return
    model_file_path = selectModelFile(using_colab)

    if using_colab:
        model_save_dir = "/content/"
    else:
        model_save_dir = os.path.join(os.getcwd(), "saved_models")
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

    dataset = input("Enter the dataset name: ")
    epochs = int(input("Enter the maximum number of epochs to finetune (default is 50): ") or 50)
    finetuner = finetune_model.Finetuner(model_state_dir=model_file_path, 
                                         finetune_csv_dir=finetune_csv_dir, 
                                         dataset = dataset, 
                                         model_save_dir=model_save_dir)
    finetuner.fineTune(max_epochs=epochs)
    finetuner.plotLosses()
    print("Model fine-tuning completed.")



def main():
    while True:
        print("Welcome to the Seq2Point Model Training and Evaluation!")
        print("1. Train a model")
        print("2. Evaluate a model")
        print("3. Fine-tune a model")
        print("4. Exit")
        choice = input("Please enter your choice (1-4): ")
        if choice == '1':
            trainModelCLI()
        elif choice == '2':
            evaluateModelCLI()
        elif choice == '3':
            fineTuneModelCLI()
        elif choice == '4':
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()