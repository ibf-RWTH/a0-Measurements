# a0-Measurements repository

Welcome to the a0-Measurements repository.  This repository provides a tool for running inference, evaluation, and measurement on models using a test dataset. The script supports various operations including prediction of segmentation masks, evaluation of metrics, and measurement of specific parameters.

# Features

* Prediction: Generate segmentation masks from models.
* Evaluation: Compute metrics to evaluate model performance.
* Measurement: Measure specific characteristics (e.g., crack size).
* Flexible Operations: Combine prediction, evaluation, and measurement as needed.

# Requirements

* Python 3.x
* Required packages (can be installed via requirements.txt if available):
  * \ albumentations==1.3.0
  * \matplotlib==3.8.0
  * \numpy==1.23.4
  * \opencv-python==4.7.0.68
  * \pandas==2.1.1
  * \patchify==0.2.3
  * \pillow==9.3.0
  * \scikit-image==0.19.3
  * \scikit-learn==1.1.3
  * \seaborn==0.13.2
  * \segmentation_models_pytorch==0.3.4
  * \torch==2.5.1
  * \torchvision==0.20.1
  * \torch-summary==1.4.5
  * \tqdm==4.67.1
* Custom module: \customevaluation from the lib folder

# Installation

Clone this repository:

git clone <repository-url>

Navigate to the project directory:

cd <repository-folder>

Install dependencies:

pip install -r requirements.txt

Usage

Run the script using the command line:

python script_name.py [--type TYPE] [--data_dir DATA_DIR] [--model_dir MODEL_DIR] [--target_dir TARGET_DIR] [--prefix PREFIX] [--target_dims TARGET_DIMS]

Arguments

--type: Type of evaluation to perform. Options:

P: Predict segmentation masks

E: Evaluate metrics (requires prediction masks)

M: Measure specific parameters (requires prediction masks)

PE: Predict and evaluate

PM: Predict and measure

EM: Evaluate and measure (requires prediction masks)

A: All types

Default: A

--data_dir: Directory where the test images and labels are stored. Default: ./data/het_test_data

--model_dir: Directory to import the models from. Default: ./models

--target_dir: Directory to store the results. Default: ./results/evaluation

--prefix: Prefix for the prediction files. Default: /test_

--target_dims: Target dimensions for the model evaluation. Default: (1024, 1024)

Example

python script_name.py --type PE --data_dir ./data/test_data --model_dir ./models --target_dir ./results

Structure

lib/customevaluation.py: Contains the ModelEvaluationAndMeasurements class for handling prediction, evaluation, and measurement tasks.

data/: Default directory for test data.

models/: Default directory for storing models.

results/: Default directory for storing evaluation results.

How It Works

Argument Parsing: The script parses command-line arguments to determine the operation type and directory paths.

Model Selection: Lists available models in the specified directory and prompts the user to select one.

Evaluation Execution: Depending on the selected type (--type), the script performs prediction, evaluation, or measurement.

Output: Results are printed and saved in the specified target directory.

License

This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments

The ModelEvaluationAndMeasurements class and methods were provided by the customevaluation module.

The tool was built to facilitate model testing and performance evaluation.

