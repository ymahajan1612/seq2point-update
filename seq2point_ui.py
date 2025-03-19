import gradio as gr
from model_pipeline.train_model import Trainer
from model_pipeline.test_model import Tester
from model_pipeline.finetune_model import Finetuner
from model_pipeline.seq2Point_factory import Seq2PointFactory
from dataset_management.dataset_registry import DatasetRegistry
import sys


def processUploadedFiles(files):
    return [file.name for file in files]

def train_model(epochs, learning_rate):
    # Placeholder function for training
    return f"Training started with {epochs} epochs and learning rate {learning_rate}"

def finetune_model(model_file, dataset):
    # Placeholder function for fine-tuning
    return f"Fine-tuning {model_file.name} on {dataset.name}"

def test_model(model_file, test_data):
    # Placeholder function for testing
    return f"Testing {model_file.name} on {test_data.name}"

with gr.Blocks() as demo:
    with gr.Tab("Train Model"):
        gr.Markdown("## Train a New Model")
        models = [model_name for model_name in Seq2PointFactory.getModelMappings().keys()]
        model_name = gr.Dropdown(label="Select Model", choices=models)
        train_csv_dirs = gr.Files(label="Upload Training Data", file_types=["csv"], type='filepath')
        validation_csv_dirs = gr.Files(label="Upload Validation Data", file_types=["csv"], type='filepath')

        dataset = gr.Dropdown(label="Select Dataset", choices=DatasetRegistry.getDatasets(), interactive=True)
        available_appliances = DatasetRegistry.getAvailableAppliances(dataset.value.lower())
        appliance = gr.Dropdown(label="Select Appliance", choices=available_appliances, interactive=True)

        running_in_colab = 'google.colab' in sys.modules
        if running_in_colab:
            model_save_dir = "/content"
        else:
            model_save_dir = '.'
        input_window_length = gr.Slider(label="Input Window Length", minimum=10, maximum=800, step=1, value=599, interactive=True)
        train_button = gr.Button("Start Training")
        train_output = gr.Textbox(label="Training Status")
        # train_button.click(train_model, inputs=[model_name, train_csv_dirs, validation_csv_dirs, appliance, dataset, model_save_dir, input_window_length], outputs=train_output)


    
    with gr.Tab("Fine-tune Model"):
        gr.Markdown("## Fine-tune an Existing Model")
        model_file = gr.File(label="Upload Model (.pth)")
        dataset = gr.File(label="Upload Dataset")
        finetune_button = gr.Button("Start Fine-tuning")
        finetune_output = gr.Textbox(label="Fine-tuning Status")
        finetune_button.click(finetune_model, inputs=[model_file, dataset], outputs=finetune_output)
    
    with gr.Tab("Test Model"):
        gr.Markdown("## Test a Model")
        test_model_file = gr.File(label="Upload Model (.pth)")
        test_data = gr.File(label="Upload Test Data")
        test_button = gr.Button("Start Testing")
        test_output = gr.Textbox(label="Testing Status")
        test_button.click(test_model, inputs=[test_model_file, test_data], outputs=test_output)

demo.launch(share=True)