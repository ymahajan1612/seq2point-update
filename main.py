# main.py
from model_pipeline import seq2Point_factory, train_model, test_model, finetune_model
import os
from IPython import get_ipython


def runningColab():
    """
    Return True if running in a Colab kernel.
    """
    try:
        import google.colab  # noqa: F401
        return get_ipython() is not None
    except ImportError:
        return False


def get_path_input(prompt: str, multiple: bool = False):
    """
    Ask the user to paste paths when GUI/file‑upload options are unavailable.
    """
    if multiple:
        paths = input(prompt).strip().split()
        return [p for p in paths if p]
    else:
        path = input(prompt).strip()
        return [path] if path else []


def selectCSVFiles(type, using_colab, *, single_file: bool = False):
    """
    Select CSV files for training / validation / testing.
    """
    print(f"Please provide the {type} CSV file{'s' if not single_file else ''}.")
    # ------------------------------------
    if using_colab:
        # ➜ Colab widget route
        from google.colab import files

        uploaded = files.upload()
        csv_files = [f"/content/{name}" for name in uploaded if name.endswith(".csv")]

        if not csv_files:
            print("No CSV files were uploaded.")
            return []

        if single_file and len(csv_files) > 1:
            print("Multiple files uploaded; using the first one.")
            csv_files = [csv_files[0]]

        return csv_files
    # ------------------------------------
    # Not running in a Colab kernel
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if single_file:
            path = filedialog.askopenfilename(
                title=f"Select {type} CSV file",
                filetypes=[("CSV files", "*.csv")],
            )
            if not path:
                # Fallback to text prompt
                return get_path_input(
                    f"Path to the {type} CSV file: ", multiple=False
                )
            return [path]
        else:
            paths = filedialog.askopenfilenames(
                title=f"Select {type} CSV files",
                filetypes=[("CSV files", "*.csv")],
            )
            if not paths:
                return get_path_input(
                    f"Paths to the {type} CSV files (space‑separated): ",
                    multiple=True,
                )
            return list(paths)

    except Exception:
        if single_file:
            return get_path_input(
                f"Path to the {type} CSV file: ", multiple=False
            )
        else:
            return get_path_input(
                f"Paths to the {type} CSV files (space‑separated): ",
                multiple=True,
            )


def selectModelFile(using_colab: bool):
    """
    Choose the .pth model file.
    """
    print("Please provide the model file (*.pth).")
    if using_colab:
        from google.colab import files

        uploaded = files.upload()
        if not uploaded:
            print("No model file uploaded.")
            return None
        model_path = list(uploaded.keys())[0]
        return f"/content/{model_path}"

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        path = filedialog.askopenfilename(
            title="Select model file", filetypes=[("Model files", "*.pth")]
        )
        if not path:
            path = input("Path to the model .pth file: ").strip()

        return path or None

    except Exception:
        path = input("Path to the model .pth file: ").strip()
        return path or None


# ---------------------------------------------------------------------
# CLI wrappers
# ---------------------------------------------------------------------
def trainModelCLI():
    print("Training a Seq2Point model ...")
    using_colab = runningColab()

    train_csv_dirs = selectCSVFiles("train", using_colab)
    validation_csv_dirs = selectCSVFiles("val", using_colab)

    if not train_csv_dirs or not validation_csv_dirs:
        print("File selection aborted. Exiting.")
        return

    appliance = input("Enter the appliance name: ")
    dataset = input("Enter the dataset name: ")

    available_models = seq2Point_factory.Seq2PointFactory.getModelMappings()
    numbered_models = {str(i + 1): m for i, m in enumerate(available_models)}

    model_save_dir = "/content" if using_colab else os.path.join(os.getcwd(), "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    print("Available models:")
    for num, model in numbered_models.items():
        print(f"  {num}: {model}")

    model_num = input("Select a model by number: ")
    input_window_length = int(input("Input window length (default 599): ") or 599)
    num_epochs = int(input("Number of epochs (default 10): ") or 10)

    trainer = train_model.Trainer(
        model_name=numbered_models[model_num],
        train_csv_dirs=train_csv_dirs,
        validation_csv_dirs=validation_csv_dirs,
        appliance=appliance,
        dataset=dataset,
        model_save_dir=model_save_dir,
        window_length=input_window_length,
    )
    trainer.trainModel(num_epochs)
    trainer.plotLosses()
    print("Model training completed.")


def evaluateModelCLI():
    print("Evaluating a Seq2Point model ...")
    using_colab = runningColab()

    test_csv_dirs = selectCSVFiles("test", using_colab, single_file=True)
    if not test_csv_dirs:
        print("No test file selected. Exiting.")
        return
    test_csv_dir = test_csv_dirs[0]

    model_file_path = selectModelFile(using_colab)
    if not model_file_path:
        print("No model file selected. Exiting.")
        return

    tester = test_model.Tester(model_state_dir=model_file_path, test_csv_dir=test_csv_dir)
    tester.testModel()
    tester.plotResults()

    mae, sae, inference_time = tester.getMetrics()
    print(f"MAE: {mae}\nSAE: {sae}\nInference time: {inference_time}")
    print("Model testing completed.")

    results = tester.getResults()
    output_name = f"{tester.appliance_name_formatted}_results.csv"
    results.to_csv(output_name, index=False)
    print(f"Results saved to {output_name}")


def fineTuneModelCLI():
    print("Fine‑tuning a Seq2Point model ...")
    using_colab = runningColab()

    finetune_csv_dirs = selectCSVFiles("finetune", using_colab, single_file=True)
    if not finetune_csv_dirs:
        print("No finetune file selected. Exiting.")
        return
    finetune_csv_dir = finetune_csv_dirs[0]

    model_file_path = selectModelFile(using_colab)
    if not model_file_path:
        print("No model file selected. Exiting.")
        return

    model_save_dir = "/content" if using_colab else os.path.join(os.getcwd(), "saved_models")
    os.makedirs(model_save_dir, exist_ok=True)

    dataset = input("Enter the dataset name: ")
    epochs = int(input("Maximum epochs to fine‑tune (default 50): ") or 50)

    finetuner = finetune_model.Finetuner(
        model_state_dir=model_file_path,
        finetune_csv_dir=finetune_csv_dir,
        dataset=dataset,
        model_save_dir=model_save_dir,
    )
    finetuner.fineTune(max_epochs=epochs)
    finetuner.plotLosses()
    print("Model fine‑tuning completed.")


# ---------------------------------------------------------------------
# Menu loop
# ---------------------------------------------------------------------
def main():
    MENU = (
        "Welcome to the Seq2Point Model Training and Evaluation!\n"
        "1. Train a model\n"
        "2. Evaluate a model\n"
        "3. Fine‑tune a model\n"
        "4. Exit"
    )
    while True:
        print(MENU)
        choice = input("Please enter your choice (1‑4): ").strip()
        if choice == "1":
            trainModelCLI()
        elif choice == "2":
            evaluateModelCLI()
        elif choice == "3":
            fineTuneModelCLI()
        elif choice == "4":
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
