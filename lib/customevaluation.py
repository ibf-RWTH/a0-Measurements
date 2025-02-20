# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:25:43 2023

@author: ros
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import torch

from matplotlib.colors import ListedColormap
from PIL import Image
from skimage.color import rgba2rgb
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from lib.customdatasets import SegmentationDataSet, SegmentationDataSetCastIron
from lib.inference_patchify import class_wise, gray2class_castiron, Predict, Metrics
from lib.measurement import Macroscale
from lib.transformations import (ComposeDouble,
                                 normalize_01,
                                 FunctionWrapperDouble,
                                 ) 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#%% custom functions
def get_filenames_of_path(path: pathlib.Path, ext: str = "*"):
    """Returns a list of files in a directory/path. Uses pathlib."""
    filenames = [file for file in path.glob(ext) if file.is_file()]
    
    return filenames

def preprocess(img: np.ndarray):
    """
    Transforms image from RGBA to RGB with additional batch dimension.
    
    Parameters
    ----------
    img : np.ndarray
        RGBA image with shape (H, W, C).

    Returns
    -------
    img : np.ndarray
        Normalized RGB image with added batch dimension and shape (B, C, H, W).

    """
    if len(img[0][0])==4:
        img = rgba2rgb(img)
    img = np.moveaxis(img, -1, 0)  # from [H, W, C] to [C, H, W]
    img = normalize_01(img)  # linear scaling to range [0-1]
    img = np.expand_dims(img, axis=0)  # add batch dimension [B, C, H, W]
    img = img.astype(np.float32)  # typecasting to float32
    
    return img

def postprocess(img: torch.tensor):
    """
    Transforms torch.tensor to numpy.array.
    
    Parameters
    ----------
    img : torch.tensor
        Image as torch.tensor.    
    
    Returns
    -------
    img : np.array
        Image as numpy.array.

    """
    img = torch.argmax(img, dim=1)  # perform argmax to generate 1 channel
    img = img.cpu().numpy()  # send to cpu and transform to numpy.ndarray
    img = np.squeeze(img)  # remove batch dim and channel dim -> [H, W]
    
    return img

def transform_image(image):
    """
    Transforms image to tensor and normalizes it.
    
    Parameters
    ----------
    image : np.array
        DESCRIPTION.

    Returns
    -------
    TYPE
        Image as numpy.array.

    """
    my_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image_aux = image
    
    return my_transforms(image_aux).unsqueeze(0) # wie np.expand() fügt neue Dimension hinzu

def colormapper(mask):
    """
    Maps the colors of the prediction mask according to the number of classes in the image.
    
    Parameters
    ----------
    mask : np.array
        Ground truth or prediction mask.

    Returns
    -------
    Colormap for mask.

    """
    red = [128/255, 0, 0, 1]
    green = [0, 128/255, 0, 1]
    yellow = [128/255, 128/255, 0, 1]
    blue = [0, 0, 128/255, 1]
    pink = [128/255, 0, 128/255, 1]
    teal = [0, 128/255, 128/255, 1]
    gray = [128/255, 128/255, 128/255, 1]
    brown = [64/255, 0, 0, 1]
    
    class_labels = np.unique(mask)
    max_class = max(class_labels)
    if max_class == 7:
        newcmp = ListedColormap([red, green, yellow, blue, pink, teal, gray, brown])
    elif max_class == 6:
        newcmp = ListedColormap([red, green, yellow, blue, pink, teal, gray])
    elif max_class == 5:
        newcmp = ListedColormap([red, green, yellow, blue, pink, teal])
    elif max_class == 4:
        newcmp = ListedColormap([red, green, yellow, blue, teal])
    elif max_class == 3:
        newcmp = ListedColormap([red, green, yellow, teal])
                 
    return newcmp

def array_saver(pred_dir, inputs, pred_prefix, outputs, cmap_req):
    """
    Saves images as png-files to pred_dir.
    
    Parameters
    ----------
    pred_dir : dir
        Directory to save the images to. Creates new dir if inexistent.
    inputs : str
        Input paths to extract the filenames from.
    pred_prefix : str
        Prefix to add to the filename.
    outputs : list of numpy.arrays 
        Are saved to pred_dir.
    cmap_req : bool
        Colormap required? e.g. True for prediction plots, False for images

    Returns
    -------
    None.

    """
    if not os.path.isdir(pred_dir):
        os.mkdir(pred_dir)
        
    for i_opt in range(len(outputs)):
        td = pred_dir + pred_prefix + inputs[i_opt].name
        if cmap_req == False:
            plt.imsave(fname=td, arr=outputs[i_opt], format = "png")
        elif cmap_req == True:
            plt.imsave(fname=td, arr=outputs[i_opt], format = "png", cmap=colormapper(outputs[i_opt]))

def list(generator):
    list_ = []
    for ele in generator:
        list_.append(ele)
    return list_

def resizefromgen(output, image_dims):
    """
    Resize an image coming from a generator to the image_dims

    Parameters
    ----------
    output : np.array
        Image to be resized.
    image_dims : tuple(int, int)
        Target dimensions.

    Yields
    ------
    m_output : np.array
        Resized image.

    """
    i=0
    for ele in output:
        pic_width, pic_height = image_dims[i]
        i+=1
        m_output = resize(ele, (pic_height, pic_width)).astype(np.uint8)
        yield m_output

#%% 
class ModelEvaluationAndMeasurements:
    """Calculate metricss for segmentation model and dataset with caching and pretransforms."""
    def __init__(self,
                 args=None, 
                 model_index=int
                 ):
        self.args = args
        self.model_index = model_index
        
        # device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        torch.manual_seed(42)
        
        # root directory
        root = pathlib.Path(self.args.data_dir)

        # input and target files
        self.inputs = get_filenames_of_path(root / "images")
        self.targets = get_filenames_of_path(root / "labels")
            
        #target dimensions of image
        self.patch_size = int(self.args.target_dims[0] / 2)

        # read images and store them in memory
        ##############################################################
        self.images_gen_PIL = (Image.open(f).convert("RGB") for f in self.inputs)
        self.image_dims_gen = (image.size for image in self.images_gen_PIL)
        self.images_gen_PIL = (Image.open(f).convert("RGB") for f in self.inputs)
        self.images_res_gen_PIL = (transforms.Resize(self.args.target_dims)(image) for image in self.images_gen_PIL)
        ##############################################################
        self.images_gen_np = (imread(img_name) for img_name in self.inputs)
        self.targets_gen_np = (imread(tar_name) for tar_name in self.targets)

        # Resize images and targets
        self.images_res_list_np = [resize(img, self.args.target_dims) for img in self.images_gen_np]
        resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
        self.targets_res_list_np = [resize(tar, self.args.target_dims, **resize_kwargs) for tar in self.targets_gen_np]

        # models
        self.model_paths = get_filenames_of_path(pathlib.Path(self.args.model_dir))
        self.model_path = pathlib.Path(self.model_paths[self.model_index]) 
        self.model_name = self.model_path.name
        print(f"Chosen model: {self.model_name}")
        # check if model folder exists in results. if not create one
        suffix = ".pth" #.pt
        if self.model_name.endswith(suffix):
            folder_name = self.model_name[:-len(suffix)]
        self.new_dir = os.path.join(self.args.target_dir, folder_name)
        os.makedirs(self.new_dir, exist_ok=True)
        print("Target directory: ", self.new_dir)
        # check if prediction folder exists. if not create one
        self.pred_dir = self.new_dir + '/predictions'
        if not os.path.isdir(self.pred_dir):
            os.mkdir(self.pred_dir)
                
        # get image numbers
        self.image_numbers = np.empty(0, str)
        for input_path in self.inputs:
            self.image_numbers = np.append(self.image_numbers, os.path.basename(input_path))
                 
        # pre-transformations
        self.pre_transforms = ComposeDouble([FunctionWrapperDouble(resize, input=True, target=False, output_shape=(self.args.target_dims[0], self.args.target_dims[1], 3)),
                                             FunctionWrapperDouble(resize, input=False, target=True, output_shape=self.args.target_dims, order=0, anti_aliasing=False, preserve_range=True), 
                                            ]
                                            )  
    #%% prediction  
    def _runprediction(self):
        # load weights to device
        if torch.cuda.is_available():
            model = torch.load(pathlib.Path.cwd() / self.model_path)
        else:
            model = torch.load(pathlib.Path.cwd() / self.model_path, map_location=torch.device('cpu')) # Load all tensors onto the CPU
        
        # model summary
        from torchsummary import summary
        summary = summary(model, (3, int(self.args.target_dims[0]/2), int(self.args.target_dims[1]/2)))
        
        # Generate prediction instances directly into a list
        pred_instances = [Predict(img, self.patch_size, model, transform_image, postprocess, self.device) for img in self.images_res_gen_PIL]
        
        # Compute outputs for each prediction instance and store them in a list
        self.output_list_np = [inst._compute_output() for inst in pred_instances]
        
        # save predictions
        array_saver(pred_dir=self.pred_dir, inputs=self.inputs, pred_prefix=self.args.prefix, outputs=self.output_list_np, cmap_req=True)
    
    #%% evaluation 
    def _runevaluation(self):
        # load predictions from folder
        if self.args.type in ["E", "EM"]: 
            preds = get_filenames_of_path(pathlib.Path(self.pred_dir))

            if self.model_name in ["castiron.pth"]:  
                dataset_test = SegmentationDataSetCastIron(inputs=self.inputs,
                                                           targets=preds,
                                                           pic_dim=self.args.target_dims,
                                                           transform=None,
                                                           use_cache=True,
                                                           pre_transform=self.pre_transforms,
                                                           )
            else: 
                dataset_test = SegmentationDataSet(inputs=self.inputs,
                                                   targets=preds,
                                                   pic_dim=self.args.target_dims,
                                                   transform=None,
                                                   use_cache=True,
                                                   pre_transform=self.pre_transforms,
                                                   )

            # dataloader training
            batch_size = len(self.inputs)
            print("Prediction masks cached.")
            dataloader_test = DataLoader(dataset=dataset_test, batch_size = batch_size, shuffle=False, num_workers = 0)
            
            x, y_pred = next(iter(dataloader_test))
            y_pred_np = y_pred.numpy()
            y_pred_np = gray2class_castiron(y_pred_np)
            output_list_np = [pred for pred in y_pred_np]
            self.output_list_np = output_list_np
            
        elif self.args.type in ["PE", "A"]:
            if not hasattr(self, 'output_list_np') or self.output_list_np is None:
                raise ValueError("output_list_np is not set. Please run `_runprediction` first.")
            output_list_np = self.output_list_np
        
        # load ground truths from folder
        if self.model_name in ["castiron.pth"]:  
            dataset_test = SegmentationDataSetCastIron(inputs=self.inputs,
                                                       targets=self.targets,
                                                       pic_dim=self.args.target_dims,
                                                       transform=None,
                                                       use_cache=True,
                                                       pre_transform=self.pre_transforms,
                                                       )
        else: 
            dataset_test = SegmentationDataSet(inputs=self.inputs,
                                               targets=self.targets,
                                               pic_dim=self.args.target_dims,
                                               transform=None,
                                               use_cache=True,
                                               pre_transform=self.pre_transforms,
                                               )
        
        # dataloader training
        batch_size = len(self.inputs)
        print("Ground truth masks cached.")
        dataloader_test = DataLoader(dataset=dataset_test, batch_size = batch_size, shuffle=False, num_workers = 0)
        
        x, y = next(iter(dataloader_test))
 
        # there is no distinction between ductile and brittle for the cast iron macro models for the other models this is done in when calling Metrics in inference_patchify
        if self.model_name in ["castiron.pth"]:
            y = gray2class_castiron(y)
        
        # Generate and store Metrics instances in a list once, to avoid reusing generators
        metrics_instances = [Metrics(self.inputs, y, output_list_np, i) for i in range(len(output_list_np))]
        
        # Generate confusion matrices
        conf_matrices = [inst._compute_confusion() for inst in metrics_instances]
        
        # Generate confusion matrix figures
        if self.model_name in ["castiron.pth"]:
            conf_matrix_figures = [inst._plot_confusion_castiron(normalized=False) for inst in metrics_instances]
        else:
            conf_matrix_figures = [inst._plot_confusion(normalized=False) for inst in metrics_instances]

        # calculate metrics     
        metrics_list = [inst._compute_metrics() for inst in metrics_instances]

        # Separate `metrics_report` into its own list
        metrics_floats = []
        metrics_reports = []
        
        for metrics in metrics_list:
            metrics_floats.append(metrics[:-1])  # All but the last value
            metrics_reports.append(metrics[-1])  # The last value (metrics_report)
        
        # Convert the floats to a NumPy array
        metrics_floats = np.array(metrics_floats)
        
        pixel_accuracy = metrics_floats[:, 0]
        macro_precision = metrics_floats[:, 1]
        macro_recall = metrics_floats[:, 2]
        macro_f1 = metrics_floats[:, 3]
        mIoU = metrics_floats[:, 4]

        avg_pixel_acc = np.mean(pixel_accuracy)
        avg_mprec = np.mean(macro_precision)
        avg_mrec = np.mean(macro_recall)
        avg_mf1 = np.mean(macro_f1)
        avg_mIoU = np.mean(mIoU)

        macro_metrics = np.array([self.image_numbers, pixel_accuracy, macro_precision, macro_recall, macro_f1, mIoU], dtype=object) #'dtype=object'
        df_macro_metrics = pd.DataFrame(macro_metrics.T, columns = ['image', 'pixel_accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'mIoU'])
        df_macro_metrics.loc[len(df_macro_metrics.index)] = ['AVERAGE', avg_pixel_acc, avg_mprec, avg_mrec, avg_mf1, avg_mIoU]
        
        # save result as xlsx
        filename = '{0}/Metrics for {1} .xlsx'.format(self.new_dir, self.model_name)
        df_macro_metrics.to_excel(filename)

        # class wise evaluation
        IoU_class_df = class_wise(self.inputs, metrics_reports, 'IoU')
        filename = '{0}/Class wise IoU for {1}.xlsx'.format(self.new_dir, self.model_name)
        IoU_class_df.to_excel(filename)
        F1_class_df = class_wise(self.inputs, metrics_reports, 'F1')
        filename = '{0}/Class wise F1-score for {1}.xlsx'.format(self.new_dir, self.model_name)
        F1_class_df.to_excel(filename)
        Prec_class_df = class_wise(self.inputs, metrics_reports, 'Precision')
        filename = '{0}/Class wise Precision for {1}.xlsx'.format(self.new_dir, self.model_name)
        Prec_class_df.to_excel(filename)
        Rec_class_df = class_wise(self.inputs, metrics_reports, 'Recall')
        filename = '{0}/Class wise Recall for {1}.xlsx'.format(self.new_dir, self.model_name)
        Rec_class_df.to_excel(filename)
        
        IoU_mean_series = IoU_class_df.iloc[-1]
        IoU_mean_series.name = self.model_name
        IoU_class_means_df = pd.DataFrame()
        IoU_class_means_df = pd.concat([IoU_class_means_df, IoU_mean_series], axis=1)
        
        if self.model_index == len(self.model_paths):
            tar_dir = str(pathlib.Path(self.args.model_dir)) + f'/{self.args.model_dir}/Evaluations/'
            filename = '{0}/Class wise IoU means TrH7 deeplabv3+.xlsx'.format(tar_dir) 
            #IoU_class_means_df.to_excel(filename)
    
    #%% measurements    
    def _runmeasurement(self):
        # load predictions from folder
        if self.args.type in ["M"]:
            preds = get_filenames_of_path(pathlib.Path(self.pred_dir))
           
            if self.model_name in ["castiron.pth"]:  
                dataset_test = SegmentationDataSetCastIron(inputs=self.inputs,
                                                           targets=preds,
                                                           pic_dim=self.args.target_dims,
                                                           transform=None,
                                                           use_cache=True,
                                                           pre_transform=self.pre_transforms,
                                                           )
            else: 
                dataset_test = SegmentationDataSet(inputs=self.inputs,
                                                   targets=preds,
                                                   pic_dim=self.args.target_dims,
                                                   transform=None,
                                                   use_cache=True,
                                                   pre_transform=self.pre_transforms,
                                                   )
            
            # dataloader training
            batch_size = len(self.inputs)
            print("Prediction masks cached.")
            dataloader_test = DataLoader(dataset=dataset_test, batch_size = batch_size, shuffle=False, num_workers = 0)
            
            x, y_pred = next(iter(dataloader_test))
            y_pred_np = y_pred.numpy()
            # there is no distinction between ductile and brittle for the cast iron macro models for the other models this is done in when calling Metrics in inference_patchify
            y_pred = gray2class_castiron(y_pred)
            
            output_list_np = [pred for pred in y_pred_np]
            self.output_list_np = output_list_np
            
        elif self.args.type in ["PM", "EM", "A"]:
            output_list_np = self.output_list_np            
        
        xlsx_path = self.args.data_dir + "/" + "specimen_overview.xlsx"
        data_df = pd.read_excel(xlsx_path, header=0)
        input_df = data_df.copy()
        input_df = input_df.reset_index(drop=True)
        # print(input_df.columns)
        nom_spec_width = list(input_df['Nominal Specimen Width'])
        nom_spec_thickness = list(input_df['Nominal Specimen Thickness'])
        specimen_type = list(input_df['Specimen type'])
        
        # load image dimensions
        image_dims_list = list(self.image_dims_gen)
        
        measurement_instances = [Macroscale(self.inputs[i], output_list_np[i], (image_dims_list[i]), nom_spec_width[i], nom_spec_thickness[i], specimen_type[i]) for i in range(len(output_list_np))]

        #%% calculate specimen dimensions
        dimensions_and_scale_list = [inst.get_dimensions_and_scale() for inst in measurement_instances]
        dimensions_and_scale = np.array(dimensions_and_scale_list)
        
        spec_orient = dimensions_and_scale[:,0]
        scales = dimensions_and_scale[:,1]
        thickness_B = dimensions_and_scale[:,2]
        netthickness_BN = dimensions_and_scale[:,3]
        width_W = dimensions_and_scale[:,4]
        starternotchlength_ak = dimensions_and_scale[:,5]

        dimensions_df = pd.DataFrame(dimensions_and_scale) #, mean_abs_deviation
        dimensions_df2 = pd.concat([dimensions_df, pd.DataFrame(self.image_numbers)], ignore_index=True, axis=1)
        dimensions_df2.columns = ['orientation', 'scale [mm/pixel]', 'thickness_B [mm]', 'netthickness_BN [mm]', 'width_W [mm]', 'starternotchlength_ak [mm]', 'Image'] 
        dimensions_df2 = dimensions_df2[['Image', 'orientation', 'scale [mm/pixel]', 'thickness_B [mm]', 'netthickness_BN [mm]', 'width_W [mm]', 'starternotchlength_ak [mm]']]
        
        filename = '{0}/Specimen dimensions for {1}.xlsx'.format(self.new_dir, self.model_name)
        dimensions_df2.to_excel(filename)    

        #%% area average method        
        aam_list = [inst.get_area_average() for inst in measurement_instances]
        aam = np.array(aam_list)
        
        aam_pixel_a0 = aam[:,0]
        aam_a0 = aam[:,1]
        aam_crack_aspect_ratio = aam[:,2]
        
        a0_avg_manual_series = input_df['a0 man']
        a0_avg_manual = np.array(a0_avg_manual_series)

        # save to xlsx
        aam_a0_difference = a0_avg_manual - aam_a0
        aam_a0_absdifference = np.abs(aam_a0_difference)
        aam_mean_absdifference = np.mean(aam_a0_absdifference)
        
        measurement_results = pd.DataFrame(data=[self.image_numbers, a0_avg_manual, scales, aam_a0, aam_a0_absdifference]) 
        measurement_results = measurement_results.T
        measurement_results.columns = ['Image', 'man a0 (avg) [mm]', 'scale [mm/pixel]', 'model a0 [mm]', 'absolute difference [mm]'] 
        
        filename = '{0}/Measurements for {1}.xlsx'.format(self.new_dir, self.model_name)  
        measurement_results.to_excel(filename)

        #save dimensions and measurements in extra file
        dimsandmeasure_df = pd.concat([dimensions_df2, pd.DataFrame(aam_a0)], ignore_index=True, axis=1)
        dimsandmeasure_df.columns = ['Image', 'orientation', 'scale [mm/pixel]', 'thickness_B [mm]', 'netthickness_BN [mm]', 'width_W [mm]', 'starternotchlength_ak [mm]', 'a0 [mm]'] 
        
        filename = '{0}/Specimen measurement for {1}.xlsx'.format(self.new_dir, self.model_name) 
        dimsandmeasure_df.to_excel(filename)    