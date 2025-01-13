# <ins>a0-Measurements repository</ins>

Welcome to the a0-Measurements repository.  This repository provides a tool for running inference, evaluation, and measurement on models using a test dataset. The script supports various operations including prediction of segmentation masks, evaluation of metrics, and measurement of specific parameters.

## Features

* Prediction: Generate segmentation masks from models using the three provided models:
  * For steel SE(B) specimens (preferably captured in front of a blue background) use: `homogeneous.pth`. 
  * For all other specimen types (and background colors) use: `heterogeneous.pth` .
  * For castiron specimens (preferably captured in front of a blue background) use: `castiron.pth`.
* Evaluation: Compute metrics to evaluate model performance. The following metrics are analyzed: 
  * Pixel Accuracy
  * Precision, Recall and F1-score
  * mIoU (mean Intersection over Union)
* Measurement: Measure initial crack sizes using the area average method.
* Flexible Operations: Combine prediction, evaluation, and measurement as needed.

## Requirements

* `Python 3.x`
* Required packages (can be installed via `requirements.txt`):
  * `albumentations==1.3.0`
  * `matplotlib==3.8.0`
  * `numpy==1.23.4`
  * `opencv-python==4.7.0.68`
  * `pandas==2.1.1`
  * `patchify==0.2.3`
  * `pillow==9.3.0`
  * `scikit-image==0.19.3`
  * `scikit-learn==1.1.3`
  * `seaborn==0.13.2`
  * `segmentation_models_pytorch==0.3.4`
  * `torch==2.5.1`
  * `torchvision==0.20.1`
  * `torch-summary==1.4.5`
  * `tqdm==4.67.1`
* Custom module: `customevaluation` from the lib folder

## Installation

Clone this repository:
```sh
git clone https://github.com/ibf-RWTH/a0-Measurements.git
```

Navigate to the project directory:
```sh
cd <repository-folder>
```

Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

Run the script using the command line:
```sh
python model_evaluation.py [--type TYPE] [--data_dir DATA_DIR] [--model_dir MODEL_DIR] [--target_dir TARGET_DIR] [--prefix PREFIX] [--target_dims TARGET_DIMS]
```

## Arguments

* `--type`: Type of evaluation to perform. Options:
  * `P`: Predict segmentation masks
  * `E`: Evaluate metrics (requires prediction masks)
  * `M`: Measure specific parameters (requires prediction masks)
  * `PE`: Predict and evaluate
  * `PM`: Predict and measure
  * `EM`: Evaluate and measure (requires prediction masks)
  * `A`: All types (default)

* `--data_dir`: Directory where the test images and labels are stored. Default: `./data/steel_test_data`.

* `--model_dir`: Directory to import the models from. Default: `./models`.

* `--target_dir`: Directory to store the results. Default: `./results/evaluation`.

* `--prefix`: Prefix for the prediction files. Default: `/test_`.

* `--target_dims`: Target dimensions for the model evaluation. Default: `(1024, 1024)`.

## Example

```sh
python model_evaluation.py --type PE --data_dir ./data/test_data --model_dir ./models --target_dir ./results
```

## Structure

* `lib/customevaluation.py`: Contains the `ModelEvaluationAndMeasurements` class for handling prediction, evaluation, and measurement tasks.

* `data/`: Default directory for test data.

* `models/`: Default directory for storing models.

* `results/`: Default directory for storing evaluation results.

## How It Works

1. **Argument Parsing**: The script parses command-line arguments to determine the operation type and directory paths.

2. **Model Selection**: Lists available models in the specified directory and prompts the user to select one.

3. **Evaluation Execution**: Depending on the selected type (`--type`), the script performs prediction, evaluation, or measurement.

4. **Output**: Results are printed and saved in the specified target directory.

## Reference / Citation

Please reference this repository as followed.
```python
@article{Rosenberger.2023,
 abstract = {Engineering Fracture Mechanics, 293 (2023) 109686. doi:10.1016/j.engfracmech.2023.109686},
 author = {Rosenberger, Johannes and Tlatlik, Johannes and M{\"u}nstermann, Sebastian},
 year = {2023},
 title = {Deep learning based initial crack size measurements utilizing macroscale fracture surface segmentation},
 keywords = {Deep Learning;Fractography;Fracture mechanics;Macroscale;Semantic Segmentation},
 pages = {109686},
 volume = {293},
 issn = {00137944},
 journal = {Engineering Fracture Mechanics},
 doi = {10.1016/j.engfracmech.2023.109686},
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

The research project “KEK Automated analysis of fracture surfaces using artificial neural networks (KNN) for nuclear relevant safety components” is funded by the German Federal Ministry for the Environment, Nature Conservation, Nuclear Safety and Consumer Protection (Project No. 1501621 ) on basis of a decision by the German Bundestag.
