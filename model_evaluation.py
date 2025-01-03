# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:25:23 2023

@author: ros
"""

# Imports
import argparse
import os
import pathlib
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# script_directory = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.dirname(os.path.abspath(__file__))

# Append the 'lib' folder to the system path
lib_path = os.path.join(script_dir, 'lib')
sys.path.append(lib_path)

# # Append it to sys.path if not already included
# if script_directory not in sys.path:
#     sys.path.append(script_directory)
# sys.path.append(r'./lib')

from lib.customevaluation import ModelEvaluationAndMeasurements 

#%%
def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Inference and measurement for all models on test-dataset.')
    # type
    parser.add_argument('--type', type=str, default='A', 
                        help='Type of evaluation: P - Predict segmentation masks, \
                                                  E - Evaluate metrics (requires prediction masks), \
                                                  M - Measure a0 (requires prediction masks), \
                                                  PE - Predict and evaluate, \
                                                  PM - Predict and measure, \
                                                  EM - Evaluate and measure (requires prediction masks), \
                                                  A - All types.'
                        )
    # data_dir 
    parser.add_argument('--data_dir', type=str, default='./data/het_test_data', 
                        help='Directory where the test images and labels are stored.')
    # model_dir
    parser.add_argument('--model_dir', type=str, default='./models', 
                        help='Directory to import the models from.')
    # model_dir
    parser.add_argument('--target_dir', type=str, default='./results/evaluation', 
                        help='Directory to import the models from.')
    # pred_prefix
    parser.add_argument('--prefix', type=str, default='/test_',
                        help='Prefix for the prediction files.')
    # target_dims
    parser.add_argument('--target_dims', type=tuple, default=(1024,1024),
                        help='Target dimensions for the model evaluation.')
    return parser.parse_args()

def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    
    return filenames

#%%
if __name__ == "__main__":
    args = get_arguments()

    model_paths = get_filenames_of_path(pathlib.Path(args.model_dir)) 
    
    # Check if the directory exists
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"The specified directory does not exist: {args.data_dir}")
    
    # Continue with your main logic if the directory exists
    print(f"Directory {args.data_dir} found. Proceeding with the operation.")

    
    for i_path in range(len(model_paths)):
        print(f"{i_path}: {model_paths[i_path]}")

    model_index = int(input("Choose a model from the above list and enter its index: "))
    
    print(f"Type-{args.type} evaluation started.")
    start_time = time.time()
    evaluator = ModelEvaluationAndMeasurements(args, model_index)
    
    if args.type in ["P", "PE", "PM", "A"]:
        evaluator._runprediction()
        print("Prediction successful.")
    if args.type in ["E", "PE", "EM", "A"]:
        evaluator._runevaluation()
        print("Metrics evaluated successful.")
    if args.type in ["M", "PM", "EM", "A"]:
        evaluator._runmeasurement()
        print("Crack size measurement successful.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Type-{args.type} evaluation successful ({elapsed_time:.3f} s).")