# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import torch
import warnings

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from patchify import patchify, unpatchify
from PIL import Image
from skimage.measure import label, regionprops
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, jaccard_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from skimage.transform import resize
from statistics import mean

#%%
def color_ticks(tick_labels):
    """
    Maps class names to colors for plots.

    Parameters
    ----------
    tick_labels : str
        Names of the classes.

    Returns
    -------
    None.

    """
    red = [128/256, 0, 0, 1]
    green = [0, 128/256, 0, 1]
    yellow = [128/256, 128/256, 0, 1]
    blue = [0, 0, 128/256, 1]
    pink = [128/256, 0, 128/256, 1]
    teal = [0, 128/256, 128/256, 1]
    gray = [128/256, 128/256, 128/256, 1]
    brown = [64/256, 0, 0, 1]
    
    newcmp = ListedColormap([red, green, yellow, blue, pink, teal, gray, brown])
    
    for tick_label in tick_labels: #https://stackoverflow.com/questions/72660993/change-seaborn-heatmap-y-ticklabels-font-color-for-alternate-labels
        if tick_label.get_text()=='background': # https://stackoverflow.com/questions/48888602/accessing-matplotlib-text-object-label-text
            tick_label.set_color(red)
        elif tick_label.get_text()=='side groove':
            tick_label.set_color(green)
        elif tick_label.get_text()=='erosion notch':
            tick_label.set_color(yellow)
        elif tick_label.get_text()=='fatigue precrack':
            tick_label.set_color(blue)
        elif tick_label.get_text()=='ductile fracture':
            tick_label.set_color(pink)
        elif tick_label.get_text()=='brittle fracture':
            tick_label.set_color(teal)
        elif tick_label.get_text()=='other':
            tick_label.set_color(gray)
        elif tick_label.get_text()=='gauge notch':
            tick_label.set_color(brown)
            
def class_IoU(inputs, metrics_report):
    """
    Calculates the class wise IoU values

    Parameters
    ----------
    inputs : str
        Input paths.
    metrics_report : pd.DataFrame
        Columns ['pixel accuracy', 'precision','recall', 'f1', 'IoU'].

    Returns
    -------
    IoU_class_df : pd.DataFrame
        Columns ['pixel accuracy', 'precision','recall', 'f1', 'IoU', + each class label in prediction mask].

    """
    # list of empty lists for class wise results
    IoU_lists = [[], [], [], [], [], [], [], []]
    
    img_numbers_list = []
    
    class_labels_list = ['0', '1', '2', '3', '4', '5', '6', '7']
    
    for m in range(len(metrics_report)):
        img_number = pathlib.Path(inputs[m]).name
        img_numbers_list.append(img_number)
        
        for cl in range(len(class_labels_list)):
            if class_labels_list[cl] in metrics_report[m].index:
                IoU_lists[cl].append(metrics_report[m].IoU[class_labels_list[cl]])
            else:
                IoU_lists[cl].append(1)

    for IoU_list in IoU_lists:
        IoU_list.append(mean(IoU_list))
  
    img_numbers_list.append('AVERAGE')
    
    #confusion matrix
    data = {'image': img_numbers_list,
            'background':IoU_lists[0], #IoU_list_background,
            'side groove': IoU_lists[1], #IoU_list_sidenotch,
            'erosion notch': IoU_lists[2], #IoU_list_erosionnotch,
            'precrack': IoU_lists[3], #IoU_list_precrack,
            'ductile fracture': IoU_lists[4], #IoU_list_ductilefracture,
            'cleavage': IoU_lists[5], #IoU_list_cleavage,
            'other': IoU_lists[6], #IoU_list_other,
            'gaugenotch': IoU_lists[7], #IoU_list_gaugenotch,
            }

    IoU_class_df = pd.DataFrame(data) 
    IoU_class_df.columns = ['images', 'background','side groove', 'erosion notch', 'fatigue precrack', 'ductile fracture', 'brittle fracture', 'other', 'gauge notch']
    
    return IoU_class_df

def class_wise(inputs, metrics_report, target_metric):
    """
    Parameters
    ----------
    inputs : WindowsPath-List
        List of WindowsPath, one entry per test image.
    metrics_report : np.array
        Contains all metrics report matrices, one entry per test image (one of the outputs of compute_metrics function). 
    target_metric : String
        Precision, Recall, F1, IoU.

    Returns
    -------
    metrics_class_df : np.array
        Class-wise metrics (1 row per test image).

    """
    # list of empty lists for class wise results
    metrics_lists = [[], [], [], [], [], [], [], []]
    
    img_numbers_list = []
    
    class_labels_list = ['0', '1', '2', '3', '4', '5', '6', '7']
    
    for m in range(len(metrics_report)):
        img_number = pathlib.Path(inputs[m]).name
        img_numbers_list.append(img_number)
        
        for cl in range(len(class_labels_list)):
            if class_labels_list[cl] in metrics_report[m].index:
                if target_metric == "IoU":
                    metrics_lists[cl].append(metrics_report[m].IoU[class_labels_list[cl]])
                elif target_metric == 'F1':
                    metrics_lists[cl].append(metrics_report[m].f1[class_labels_list[cl]])
                elif target_metric == 'Precision':
                    metrics_lists[cl].append(metrics_report[m].precision[class_labels_list[cl]])
                elif target_metric == 'Recall':
                    metrics_lists[cl].append(metrics_report[m].recall[class_labels_list[cl]])
            else:
                metrics_lists[cl].append(1)

    for metrics_list in metrics_lists:
        metrics_list.append(mean(metrics_list))
  
    img_numbers_list.append('AVERAGE')
    
    #confusion matrix
    data = {'image': img_numbers_list,
            'background':metrics_lists[0], #IoU_list_background,
            'side groove': metrics_lists[1], #IoU_list_sidenotch,
            'erosion notch': metrics_lists[2], #IoU_list_erosionnotch,
            'precrack': metrics_lists[3], #IoU_list_precrack,
            'ductile fracture': metrics_lists[4], #IoU_list_ductilefracture,
            'cleavage': metrics_lists[5], #IoU_list_cleavage,
            'other': metrics_lists[6], #IoU_list_other,
            'gaugenotch': metrics_lists[7], #IoU_list_gaugenotch,
            }

    metrics_class_df = pd.DataFrame(data) 
    metrics_class_df.columns = ['images', 'background','side groove', 'erosion notch', 'fatigue precrack', 'ductile fracture', 'brittle fracture', 'other', 'gauge notch']
    
    return metrics_class_df

def gray2class(patch):
    """
    Maps the grayscale values to labels.

    Parameters
    ----------
    patch : np.array
        Grayscale prediction mask.

    Returns
    -------
    patch : np.array
        Prediction mask.

    """
    patch=np.where(patch==27,0,patch) #background
    patch=np.where(patch==38,0,patch) #background
    patch=np.where(patch==91,1,patch) #side groove
    patch=np.where(patch==75,1,patch) #side groove
    patch=np.where(patch==118,2,patch) #erosion notch
    patch=np.where(patch==113,2,patch) #erosion notch
    patch=np.where(patch==9,3,patch) #fatigue precrack / starter notch
    patch=np.where(patch==15,3,patch) #fatigue precrack / starter notch
    patch=np.where(patch==53,4,patch) #ductile fracture
    patch=np.where(patch==100,5,patch) #brittle fracture
    patch=np.where(patch==90,5,patch) #brittle fracture
    """ MIT VORSICHT ZU BEHANDELN --> """
    # patch=np.where(patch==5,4,patch) because there is no distinction between ductile and brittle for the cast iron macro models
    """ <-- MIT VORSICHT ZU BEHANDELN """
    patch=np.where(patch==128,6,patch) #other
    patch=np.where(patch==19,7,patch) #8th class

    return patch

def gray2class_castiron(patch):
    """
    Maps the grayscale values to labels.

    Parameters
    ----------
    patch : np.array
        Grayscale prediction mask.

    Returns
    -------
    patch : np.array
        Prediction mask.

    """
    patch=np.where(patch==27,0,patch) #background
    patch=np.where(patch==38,0,patch) #background
    patch=np.where(patch==91,1,patch) #side groove
    patch=np.where(patch==75,1,patch) #side groove
    patch=np.where(patch==118,2,patch) #erosion notch
    patch=np.where(patch==113,2,patch) #erosion notch
    patch=np.where(patch==9,3,patch) #fatigue precrack / starter notch
    patch=np.where(patch==15,3,patch) #fatigue precrack / starter notch
    patch=np.where(patch==53,4,patch) #ductile fracture
    patch=np.where(patch==100,5,patch) #brittle fracture
    patch=np.where(patch==90,5,patch) #brittle fracture
    """ MIT VORSICHT ZU BEHANDELN --> """
    patch=np.where(patch==5,4,patch) # there is no distinction between ductile and brittle for the cast iron macro models
    """ <-- MIT VORSICHT ZU BEHANDELN """
    patch=np.where(patch==128,6,patch) #other
    patch=np.where(patch==19,7,patch) #8th class

    return patch


def class2gray(patch):
    """
    Maps the labels to grayscale values and transforms it to the PIL.Image

    Parameters
    ----------
    patch : np.array
        Patch of the prediction mask.

    Returns
    -------
    pil_patch : PIL.Image
        Grayscale PIL.Image.

    """
    patch=np.where(patch==0,0,patch)
    patch=np.where(patch==1,84,patch)
    patch=np.where(patch==2,91,patch)
    patch=np.where(patch==3,112,patch)
    patch=np.where(patch==4,135,patch)
    patch=np.where(patch==5,201,patch)
    patch=np.where(patch==6,218,patch)
    patch=np.where(patch==7,220,patch)
    
    pil_patch = Image.fromarray(np.uint8(patch))
    return pil_patch


class Predict:
    def __init__(self, 
                 img,
                 patch_size,
                 model,
                 preprocess,
                 postprocess,
                 device,
                 ):
        
        self.img = img
        self.patch_size = patch_size
        self.model = model
        self.preprocess = preprocess
        self.postprocess = postprocess
        self.device = device
    
    def _compute_output(self):
        """
        For the evaluation of macro images. First the image is patchified and after prediction the patches are stitched back together.

        Returns
        -------
        output : np.array
            Prediction mask.

        """
        self.model.eval() #call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
        self.img = self.preprocess(self.img)  # preprocess image
        
        input = torch.tensor(self.img)
        img_ch = len(input[0]) #returns number of input channels
        X = np.array(input.permute(0,2,3,1)) #change shape of input from [N,C,H,W] to [N,H,W,C]
        patch_step_size = len(X[0])-self.patch_size
        X_patches = [patchify(X[i],(self.patch_size,self.patch_size,img_ch),step=patch_step_size) for i in range(len(X))] #create patches
        X_patches = np.array(X_patches) #transform List to numpy array
        input_patches = torch.tensor(X_patches) #torch.Size([N, 2, 2, 1, 512, 512, 3]) 
        input_patches = input_patches.permute(0,1,2,3,6,4,5) #torch.Size([N, 2, 2, 1, 3, 512, 512]) #create torch.tensors and change shape of patches from [N,H,W,C] to [N,C,H,W]

        out = []
        for patch_idx1 in range(len(input_patches[0])):
            for patch_idx2 in range(len(input_patches[0][patch_idx1])):
                input_patch = input_patches[0][patch_idx1][patch_idx2] #torch.Size([1, 3, 512, 512])
                input_patch = input_patch.to(self.device)  # to torch, send to device
                
                with torch.no_grad():
                    out_patch = self.model(input_patch)  # send through model/network
                
                out_softmax = torch.softmax(out_patch, dim=1)  # perform softmax on outputs
                output_patch = self.postprocess(out_softmax)  # postprocess outputs
                out.append(output_patch)
                
        patch_array=np.array([[out[0],out[1]],[out[2],out[3]]]) # create a [2,2] matrix out of the output_patch-list in order to perform unpatchify
        output = unpatchify(patch_array, (1024, 1024))
        output = self._clean_outliers(output)
        
        return output
    
    # method ensures only the largest region (should be the specimen) is returned and therefore cleans any misdetections or outliers 
    def _clean_outliers(self, prediction_mask):
        """
        All but the biggest region in the prediction mask are deleted. This postprocessing step increases mIoU in images where background is misclassified.

        Parameters
        ----------
        prediction_mask : np.array
            Prediction mask.

        Returns
        -------
        largest_region_mask : np.array
            Postprocessed prediction mask.

        """
        binary_mask = (np.where(prediction_mask > 0, 1, prediction_mask)).astype(np.uint8)
        # Create unique labels for connected components (otherwise all 1s would be treated as one region)
        sklabeled_mask = label(binary_mask)
        # Calculate region properties to find the largest region
        regions = regionprops(sklabeled_mask)
        
        # Find the largest region based on area
        largest_region = max(regions, key=lambda r: r.area)
        largest_label = largest_region.label
        
        # Create a new mask keeping only the largest region
        largest_region_mask = np.where(sklabeled_mask == largest_label, prediction_mask, 0)
        # For some reasons some masks are returned as zeros only --> to prevent this return original mask
        # (causes error when calling customevaluations colormapper variable newcmp --> therefore set limit to 5)
        if len(np.unique(largest_region_mask))==1:
            largest_region_mask = prediction_mask
        
        return largest_region_mask    

    def _compute_output_full(self):
        """
        For the evaluation of full-scale macro images.

        Returns
        -------
        output : np.array
            Prediction mask.

        """
        self.model.eval() #call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference
        self.img = self.preprocess(self.img)  # preprocess image

        input_ = torch.tensor(self.img)
        input_ = input_.to(self.device)
        with torch.no_grad():
            out = self.model(input_)
        out_softmax = torch.softmax(out, dim=1)
        output = self.postprocess(out_softmax)
        
        resize_kwargs = {'order': 0, 'anti_aliasing': False, 'preserve_range': True}
        output = resize(output, (1024,1024), **resize_kwargs) #https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
        
        return output

class Metrics:
    def __init__(self, 
                 input_path, 
                 mask, 
                 output, 
                 idx
                 ):
        self.input_path = input_path
        self.mask = mask
        self.output = output
        self.idx = idx      
        print(self.input_path[self.idx])
        y_true = self.mask[self.idx]
        # print("Are all ground truth classes recognized correctly or is there a mixup? ", np.unique(y_true, return_counts=True))
        y_true = gray2class(y_true)
        y_pred = self.output[self.idx]
        # print("Are all predicted classes recognized correctly or is there a mixup? ", np.unique(y_pred, return_counts=True))

        
        # tensor to nparray
        y_true_np = y_true
        if torch.is_tensor(y_true_np)==True:
            y_true_np = y_true.numpy() 

        #change from shape (512, 512) to (1, 262144)
        self.y_true_flat = y_true_np.flat
        self.y_pred_flat = y_pred.flat 
        
        #sklearn confusion_matrix (pure np.array without headers or support/margins)
        self.np_conf_matrix = confusion_matrix(self.y_true_flat, self.y_pred_flat)
        
        #confusion matrix
        data = {'y_Actual':self.y_true_flat,
                'y_Predicted': self.y_pred_flat
                }
    
        df = pd.DataFrame(data, columns=['y_Actual','y_Predicted'])
        self.pd_conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'], margins = True)

    def _compute_confusion(self):
        return self.np_conf_matrix, self.pd_conf_matrix

    def _plot_confusion(self, normalized=False):
        """
        Display the confusion matrices (for cast iron use _plot_confusion_castiron). The prediction masks in comparison to the ground truth.

        Parameters
        ----------
        normalized : bool, optional
            Whether or not to normalize the confusion matrix entries. The default is False.

        Returns
        -------
        fig : matplotlib.pyplot.fig
            Plot stored as fig.

        """
        if not normalized:
            size_tpl=(10,10)
        else:
            size_tpl=(36,16)
        size_tpl=22.5,10
        fig, axes = plt.subplots(ncols=2, figsize=size_tpl, dpi=100, gridspec_kw={'width_ratios': [1, 1.25]})
        conf_plot1, conf_plot2 = axes
        img_number = pathlib.Path(self.input_path[self.idx]).name
        kwargs={'linespacing': 1.0} #, 'fontsize': 25 , 'size': 20
        plt.rcParams.update({'font.size': 20})
        #sns.set(font_scale=2)
        if len(self.np_conf_matrix)==5:
            label_list=['background', 'side groove', 'erosion notch', 'fatigue precrack', 'brittle fracture'] #'ductile fracture', 
        elif len(self.np_conf_matrix)==6:
            label_list=['background', 'side groove', 'erosion notch', 'fatigue precrack', 'ductile fracture', 'brittle fracture'] 
        elif len(self.np_conf_matrix)==7:
            label_list=['background', 'side groove', 'erosion notch', 'fatigue precrack', 'ductile fracture', 'brittle fracture', 'other']
        elif len(self.np_conf_matrix)==8:
            label_list=['background', 'side groove', 'erosion notch', 'fatigue precrack', 'ductile fracture', 'brittle fracture', 'other', 'gauge notch']        
        
        sns.heatmap(self.np_conf_matrix, annot=True, fmt="d", annot_kws={**kwargs}, linewidth=1, linecolor='black', square=False, cmap=ListedColormap(['white']), cbar=False, xticklabels=label_list, yticklabels=label_list, ax=conf_plot1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            conf_matrix_norm = self.np_conf_matrix.astype('float') / self.np_conf_matrix.sum(axis=1)[:, np.newaxis]
            sns.heatmap(conf_matrix_norm, annot=True, fmt=".4f", annot_kws={**kwargs}, linewidth=1, linecolor='w', square=False, mask=False, vmin=0.0, vmax=1.0, cmap="YlGnBu", xticklabels=label_list, yticklabels=False, ax=conf_plot2)
    
            conf_plot1.set_xlabel('Predicted Labels')
            conf_plot2.set_xlabel('Predicted Labels')
            conf_plot1.set_ylabel('Ground Truth Labels')
        
            # coloring the ticks according to labels
            color_ticks(conf_plot1.axes.get_xticklabels())
            color_ticks(conf_plot2.axes.get_xticklabels())
            color_ticks(conf_plot1.axes.get_yticklabels())
      
        fig.suptitle('{} - Confusion Matrix for {} \n\n'.format(self.idx, img_number)) #, fontdict={'fontsize': 10}

        fig = plt.gcf()
        
        plt.show()
        return fig
    
    def _plot_confusion_castiron(self, normalized=False):
        """
        Display the confusion matrices for castiron. The prediction masks in comparison to the ground truth.

        Parameters
        ----------
        normalized : bool, optional
            Whether or not to normalize the confusion matrix entries. The default is False.

        Returns
        -------
        fig : matplotlib.pyplot.fig
            Plot stored as fig.

        """
        if not normalized:
            size_tpl=(10,10)
        else:
            size_tpl=(36,16)
        size_tpl=22.5,10
        fig, axes = plt.subplots(ncols=2, figsize=size_tpl, dpi=100, gridspec_kw={'width_ratios': [1, 1.25]})
        conf_plot1, conf_plot2 = axes
        img_number = pathlib.Path(self.input_path[self.idx]).name
        kwargs={'linespacing': 1.0} 
        plt.rcParams.update({'font.size': 20})
        
        if len(self.np_conf_matrix)==5:
            label_list=['background', 'side groove', 'erosion notch', 'fatigue precrack', 'ductile fracture'] 
        elif len(self.np_conf_matrix)==4:
            label_list=['background', 'erosion notch', 'fatigue precrack', 'ductile fracture'] 
        elif len(self.np_conf_matrix)==3:
            label_list=['background', 'erosion notch', 'ductile fracture']
        
        sns.heatmap(self.np_conf_matrix, annot=True, fmt="d", annot_kws={**kwargs}, linewidth=1, linecolor='black', square=False, cmap=ListedColormap(['white']), cbar=False, xticklabels=label_list, yticklabels=label_list, ax=conf_plot1)
        conf_matrix_norm = self.np_conf_matrix.astype('float') / self.np_conf_matrix.sum(axis=1)[:, np.newaxis]
        sns.heatmap(conf_matrix_norm, annot=True, fmt=".4f", annot_kws={**kwargs}, linewidth=1, linecolor='w', square=False, mask=False, vmin=0.0, vmax=1.0, cmap="YlGnBu", xticklabels=label_list, yticklabels=False, ax=conf_plot2)

        conf_plot1.set_xlabel('Predicted Labels')
        conf_plot2.set_xlabel('Predicted Labels')
        conf_plot1.set_ylabel('Ground Truth Labels')
        
        # coloring the ticks according to labels
        color_ticks(conf_plot1.axes.get_xticklabels())
        color_ticks(conf_plot2.axes.get_xticklabels())
        color_ticks(conf_plot1.axes.get_yticklabels())
      
        fig.suptitle('{} - Confusion Matrix for {} \n\n'.format(self.idx, img_number)) 

        fig = plt.gcf()

        return fig

    def _save_confusion_plots(self, target_path, figures, model_name, normalized=False):
        """
        Save the confusion matrices as pdf.

        Parameters
        ----------
        target_path : str
            Dir to store plots in.
        figures : matplotlib.pyplot.fig
            Plot.
        model_name : str
            Model name.
        normalized : bool, optional
            Whether or not to normalize the confusion matrix entries. The default is False.

        Returns
        -------
        None.

        """
        if not normalized:
            pdf_name = '{0}/{1} Confusion Matrices.pdf'.format(target_path, model_name)
        else:
            pdf_name = '{0}/{1} Normalized Confusion Matrices.pdf'.format(target_path, model_name)
        
        with PdfPages(pdf_name) as pdf:
            for fig in figures:
                plt.tight_layout()
                pdf.savefig(fig, dpi=1000, bbox_inches='tight')
                plt.close()

    def _compute_metrics(self):    
        """
        Returns the metrics for every prediction mask.

        Returns
        -------
        pixel_accuracy : np.array
            Caculated based on sklearn.metrics.accuracy_score with average = "macro".
        macro_precision : np.array
            Caculated based on sklearn.metrics.precision_score with average = "macro".
        macro_recall : np.array
            Caculated based on sklearn.metrics.recall_score with average = "macro"..
        macro_f1 : np.array
            Caculated based on sklearn.metrics.f1_score with average = "macro"..
        mIoU : np.array
            Caculated based on sklearn.metrics.jaccard_score.
        metrics_report : pd.DataFrame
            Columns ['pixel accuracy', 'precision','recall', 'f1', 'IoU'].
        classification : np.array
            Caculated based on sklearn.metrics.classification_report.
        f1 : np.array
            Caculated based on sklearn.metrics.f1_score with average = None.
        f1_and_macro : np.array
            Caculated based on sklearn.metrics.f1_score with average = 'macro'.
        precision : np.array
            Caculated based on sklearn.metrics.precision_score with average = None.
        recall : np.array
            Caculated based on sklearn.metrics.recall_score with average = None.
        average_precision : np.array
            Caculated based on sklearn.metrics.precision_score with average = None.
        n_pred_classes : int
            Number of predicted classes in the prediction mask.

        """
        # column and row headers  
        pred_classes = np.unique(self.y_pred_flat, return_counts=True)
        true_classes = np.unique(self.y_true_flat, return_counts=True) #required for support
        
        # pixel accuracy
        pixel_accuracy = accuracy_score(self.y_true_flat, self.y_pred_flat)
        pixel_accuracy = np.round(pixel_accuracy, 5)
        
        # np.uniques of prediction and ground truth are merged --> length of index list is same for all specimens
        row_headers = sorted(list(set(list(map(str, true_classes[0]))) | set(list(map(str, pred_classes[0])))))
        row_headers = np.append(row_headers, 'macro average')
        print('Metrics ready for index ' + str(self.idx))
        
        # precision for each class
        precision=precision_score(self.y_true_flat, self.y_pred_flat, average=None)
        # macro precision and constructor for metrics_report
        macro_precision = precision_score(self.y_true_flat, self.y_pred_flat, average='macro')
        precision_and_macro = np.append(precision, macro_precision)
        precision_and_macro = np.round(precision_and_macro, 5)
        
        emptiness = []
        for i in range(len(row_headers)-1): # -1 because last row are averages
             empty = ['']
             emptiness = np.append(emptiness, empty)
        accuracy_array = np.append(emptiness, pixel_accuracy)
        
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UndefinedMetricWarning  # Suppress only UndefinedMetricWarning
            )
            # recall for each class
            recall=recall_score(self.y_true_flat, self.y_pred_flat, average=None)
            # macro recall and constructor for metrics_report
            macro_recall = recall_score(self.y_true_flat, self.y_pred_flat, average='macro')
            recall_and_macro = np.append(recall, macro_recall)
            recall_and_macro = np.round(recall_and_macro, 5)
            
            # f1-score for each class
            f1 = f1_score(self.y_true_flat, self.y_pred_flat, average=None)
            # macro f1-score and constructor for metrics_report
            # macro f1-score doesn't take class imbalance into account!
            macro_f1 = f1_score(self.y_true_flat, self.y_pred_flat, average='macro')
            f1_and_macro = np.append(f1, macro_f1)
            f1_and_macro = np.round(f1_and_macro, 5)
    
            # jaccard/IoU for each class
            IoU=jaccard_score(self.y_true_flat, self.y_pred_flat, average=None)
            # mean jaccard/IoU and constructor for metrics_report
            mIoU = jaccard_score(self.y_true_flat, self.y_pred_flat, average='macro')
            IoU_and_mIoU = np.append(IoU, mIoU)
            IoU_and_mIoU = np.round(IoU_and_mIoU, 5)
        
            metrics = pd.DataFrame([accuracy_array, precision_and_macro, recall_and_macro, f1_and_macro, IoU_and_mIoU], dtype=object) 
            metricsdf = metrics.T
            metricsdf.columns=['pixel accuracy', 'precision','recall', 'f1', 'IoU']
            print(metricsdf)
            metricsdf.index=row_headers
            metrics_report = metricsdf
            print('Metrics report calculated')
            #convert classes[0] from int to str
            classification = classification_report(self.y_true_flat, self.y_pred_flat, digits = 5) 
        
        # Warning that can be uncommented for in-detail information.
        # if len(true_classes[0]) != len(pred_classes[0]):
        #     print("UndefinedMetricWarning for index " + str(self.idx) + " - Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.")
        
        # For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        Y_test = label_binarize(self.y_true_flat, classes=[0,1,2,3,4,5,6])
        y_score = label_binarize(self.y_pred_flat, classes=[0,1,2,3,4,5,6])
        n_pred_classes = y_score.shape[1]
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="No positive class found in y_true, recall is set to one for all thresholds.",
                category=UserWarning
            )
            for i in range(n_pred_classes):
    
                precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
                average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])
            
        average_precision["micro"] = average_precision_score(Y_test, y_score, average='micro')
        precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())

        return pixel_accuracy, macro_precision, macro_recall, macro_f1, mIoU, metrics_report#, classification, f1, f1_and_macro, precision, recall, average_precision, n_pred_classes
