# Imports
import cv2
import numpy as np
import os
import torch

from patchify import patchify
from PIL import Image
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.util import img_as_ubyte
from torch.utils import data
from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#%%
class SegmentationDataSet(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 pic_dim: tuple,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.pic_dim = pic_dim
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = imread(str(img_name)), imread(str(tar_name))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        #from float to int --> Necessary for cv2.cvtColor to work
        y = y.astype('uint8')
        
        #change shape from [N,H,W,4] to [N,H,W,1]
        y = cv2.cvtColor(src=y, dst=y, code=cv2.COLOR_RGBA2GRAY)
        
        y=np.where(y==38,0,y)
        y=np.where(y==75,1,y)
        y=np.where(y==113,2,y)
        y=np.where(y==15,3,y)
        y=np.where(y==53,4,y)
        y=np.where(y==90,5,y)
        y=np.where(y==128,6,y)
        y=np.where(y==19,7,y)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
    
        return x, y

class SegmentationDataSetCastIron(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 pic_dim: tuple,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.pic_dim = pic_dim
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = imread(str(img_name)), imread(str(tar_name))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        #from float to int --> Necessary for cv2.cvtColor to work
        y = y.astype('uint8')
        
        #change shape from [N,H,W,4] to [N,H,W,1]
        y = cv2.cvtColor(src=y, dst=y, code=cv2.COLOR_RGBA2GRAY)
        
        y=np.where(y==27,0,y)
        y=np.where(y==91,1,y)
        y=np.where(y==118,2,y)
        y=np.where(y==9,3,y)
        y=np.where(y==100,4,y)
        
        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
    
        return x, y
    
class LabelRevision(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 pic_dim: tuple,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.pic_dim = pic_dim
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = imread(str(img_name)), np.asarray(Image.open(str(tar_name)).convert("RGB"))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        return x, y
    

class ClassEncodingRevision(data.Dataset):
    """Image segmentation dataset with caching and pretransforms."""
    def __init__(self,
                 inputs: list,
                 targets: list,
                 pic_dim: tuple,
                 transform=None,
                 use_cache=False,
                 pre_transform=None,
                 ):
        self.inputs = inputs
        self.targets = targets
        self.pic_dim = pic_dim
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

        if self.use_cache:
            self.cached_data = []

            progressbar = tqdm(range(len(self.inputs)), desc='Caching')
            for i, img_name, tar_name in zip(progressbar, self.inputs, self.targets):
                img, tar = imread(str(img_name)), imread(str(tar_name))
                if self.pre_transform is not None:
                    img, tar = self.pre_transform(img, tar)

                self.cached_data.append((img, tar))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):
        if self.use_cache:
            x, y = self.cached_data[index]
        else:
            # Select the sample
            input_ID = self.inputs[index]
            target_ID = self.targets[index]

            # Load input and target
            x, y = imread(str(input_ID)), imread(str(target_ID))

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        #from float to int --> Necessary for cv2.cvtColor to work
        y = y.astype('uint8')
        
        #change shape from [N,H,W,4] to [N,H,W,1]
        y = cv2.cvtColor(src=y, dst=y, code=cv2.COLOR_RGBA2GRAY)

        # Typecasting
        x, y = torch.from_numpy(x).type(self.inputs_dtype), torch.from_numpy(y).type(self.targets_dtype)
        
        
        return x, y