# Imports
import albumentations as A
import numpy as np

from skimage.util import img_as_ubyte
from skimage.util import crop
from typing import List, Callable, Tuple

#%%
def normalize_01(inp: np.ndarray):
    """
    Normalize the input array to the range [0, 1].

    This function scales the values in the input array such that the minimum
    value becomes 0 and the maximum value becomes 1. The transformation is 
    linear and does not clip any values, ensuring that the entire range of 
    the input data is preserved.

    Parameters:
    ----------
    inp : np.ndarray
        The input array to normalize. It can be of any shape and contain any 
        numerical data.

    Returns:
    -------
    np.ndarray
        A new array with the same shape as the input, where all values are 
        linearly scaled to lie within the range [0, 1].

    Example:
    --------
    >>> inp = np.array([[1, 2, 3], [4, 5, 6]])
    >>> normalize_01(inp)
    array([[0. , 0.2, 0.4],
           [0.6, 0.8, 1. ]])
    
    Notes:
    ------
    - The function uses `np.min` to find the minimum value and `np.ptp` 
      (peak-to-peak range) to compute the range of values in the array.
    - If the input array has a uniform value (e.g., all zeros), the function 
      may return NaN values due to division by zero.
    """
    inp_out = (inp - np.min(inp)) / np.ptp(inp)
    
    return inp_out

def normalize(inp: np.ndarray, mean: float, std: float):
    """
    Normalize an input array using a specified mean and standard deviation.

    This function scales the values in the input array by subtracting the 
    given mean and dividing by the given standard deviation. The resulting 
    array will have a mean of approximately 0 and a standard deviation of 1 
    if the input data initially followed the specified mean and std.

    Parameters:
    ----------
    inp : np.ndarray
        The input array to normalize. It can be of any shape and contain any 
        numerical data.
    mean : float
        The mean value to subtract from the input data.
    std : float
        The standard deviation value to divide the input data by.

    Returns:
    -------
    np.ndarray
        A new array with the same shape as the input, where all values have 
        been normalized based on the specified mean and standard deviation.

    Example:
    --------
    >>> inp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> mean = 3.0
    >>> std = 1.0
    >>> normalize(inp, mean, std)
    array([-2., -1.,  0.,  1.,  2.])
    
    Notes:
    ------
    - This function does not validate whether `std` is zero. If `std` is zero,
      a division by zero will occur, resulting in NaN or infinite values.
    - Commonly used in preprocessing for machine learning, especially for 
      normalizing inputs to have zero mean and unit variance.
    """
    inp_out = (inp - mean) / std
    
    return inp_out

def create_dense_target(tar: np.ndarray):
    """
    Convert an array of arbitrary labels into a dense, zero-based format.
    
    This function maps unique labels in the input array to consecutive integers
    starting from 0. It is particularly useful for preprocessing target labels
    in machine learning tasks where dense, zero-based labels are required.
    
    Parameters:
    ----------
    tar : np.ndarray
        An array of target labels (e.g., class labels). The labels can be
        any hashable and sortable type, such as integers or strings.
    
    Returns:
    -------
    np.ndarray
        An array of the same shape as the input, where each label in `tar` 
        is replaced by a corresponding dense integer label.
    
    Example:
    --------
    >>> tar = np.array([100, 200, 100, 300, 200])
    >>> create_dense_target(tar)
    array([0, 1, 0, 2, 1])
    
    Notes:
    ------
    The mapping from original to dense labels is determined by the order
    of unique labels in the input, sorted in ascending order.
    
    """
    classes = np.unique(tar)
    dummy = np.zeros_like(tar)
    for idx, value in enumerate(classes):
        mask = np.where(tar == value)
        dummy[mask] = idx

    return dummy

def center_crop_to_size(x: np.ndarray,
                        size: Tuple,
                        copy: bool = False,
                        ) -> np.ndarray:
    """
    Center crop an input array to the specified size.

    This function crops the input array `x` symmetrically around its center to match
    the specified spatial dimensions in `size`. It assumes that the spatial dimensions 
    of the input are even to ensure precise cropping.

    Parameters:
    ----------
    x : np.ndarray
        The input array to be cropped. It can have any shape, but the function expects 
        spatial dimensions (e.g., height and width) to be even for accurate cropping.
    size : Tuple
        The target size for the cropped array. It should match the spatial dimensions 
        of `x` in shape and order (e.g., `(height, width)` for 2D spatial arrays).
    copy : bool, optional
        If `True`, a copy of the cropped array is returned. If `False` (default), 
        a view into the original array is returned, depending on the behavior of `crop`.

    Returns:
    -------
    np.ndarray
        A new array that is center-cropped to the specified `size`.

    Example:
    --------
    >>> x = np.ones((8, 8))
    >>> size = (4, 4)
    >>> center_crop_to_size(x, size)
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])
    
    Notes:
    ------
    - The cropping parameters are computed as half of the difference between 
      the input and target sizes along each spatial dimension.
    - Ensure that the input spatial dimensions are even; otherwise, the function
      may not crop symmetrically, or it may raise errors depending on the `crop` implementation.
    """
    x_shape = np.array(x.shape)
    size = np.array(size)
    params_list = ((x_shape - size) / 2).astype(np.int).tolist()
    params_tuple = tuple([(i, i) for i in params_list])
    cropped_image = crop(x, crop_width=params_tuple, copy=copy)
    
    return cropped_image

def re_normalize(inp: np.ndarray,
                 low: int = 0,
                 high: int = 255
                 ):
    """
    Normalize the input array to a specified range. Default range: [0, 255].

    This function rescales the input array to fit within the range defined by 
    `low` and `high`. By default, it normalizes the data to an 8-bit unsigned 
    integer format (0 to 255) using `img_as_ubyte`.

    Parameters:
    ----------
    inp : np.ndarray
        The input array to be normalized. It can have any numerical data type 
        or shape but is expected to represent image data.
    low : int, optional
        The lower bound of the target range. Default is 0.
    high : int, optional
        The upper bound of the target range. Default is 255.

    Returns:
    -------
    np.ndarray
        A new array with the same shape as the input, where values are normalized 
        to the specified range.

    Example:
    --------
    >>> inp = np.array([[0.1, 0.2], [0.8, 1.0]])
    >>> re_normalize(inp)
    array([[ 25,  51],
           [204, 255]], dtype=uint8)

    Notes:
    ------
    - The function uses `img_as_ubyte` from `skimage` to normalize the array 
      to the range [0, 255], which implicitly scales values and converts the 
      data type to `uint8`.
    - Ensure that the input array values are within the expected range for 
      `img_as_ubyte` (e.g., [0, 1] for float input). Values outside this range 
      may cause clipping.
    """
    inp_out = img_as_ubyte(inp)
    
    return inp_out

def random_flip(inp: np.ndarray, tar: np.ndarray, ndim_spatial: int):
    """
    Apply a random spatial flip to input and target arrays along specified dimensions.

    This function randomly flips the input (`inp`) and target (`tar`) arrays along 
    spatial dimensions. The flipping axes are determined randomly for each spatial 
    dimension independently, with a 50% chance for each dimension to be flipped. 
    The input and target arrays may have different dimensionalities, and the flipping 
    is applied accordingly.

    Parameters:
    ----------
    inp : np.ndarray
        The input array to be flipped. Typically used for image or volumetric data, 
        with spatial dimensions starting from axis 1.
    tar : np.ndarray
        The target array to be flipped. It typically shares some spatial axes with 
        `inp` but can have fewer or different axes (e.g., for segmentation masks or labels).
    ndim_spatial : int
        The number of spatial dimensions to consider for flipping. For example, 
        use 2 for 2D spatial data or 3 for 3D volumetric data.

    Returns:
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the randomly flipped input (`inp_flipped`) and target 
        (`tar_flipped`) arrays.

    Example:
    --------
    >>> inp = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> tar = np.array([[1, 0], [0, 1]])
    >>> random_flip(inp, tar, ndim_spatial=2)
    (array([[[3, 4], [1, 2]], [[7, 8], [5, 6]]]), 
     array([[0, 1], [1, 0]]))  # Example result, actual flipping may vary.

    Notes:
    ------
    - The spatial axes of `inp` start from axis 1, while for `tar` they start from axis 0.
      Ensure `inp` and `tar` align correctly in terms of their spatial dimensions.
    - This function assumes spatial dimensions are contiguous, and `ndim_spatial` 
      does not exceed the number of dimensions in the input arrays.
    """
    flip_dims = [np.random.randint(low=0, high=2) for dim in range(ndim_spatial)]

    flip_dims_inp = tuple([i + 1 for i, element in enumerate(flip_dims) if element == 1])
    flip_dims_tar = tuple([i for i, element in enumerate(flip_dims) if element == 1])

    inp_flipped = np.flip(inp, axis=flip_dims_inp)
    tar_flipped = np.flip(tar, axis=flip_dims_tar)

    return inp_flipped, tar_flipped

#%%
class Repr:
    """Evaluable string representation of an object"""

    def __repr__(self): return f'{self.__class__.__name__}: {self.__dict__}'

class FunctionWrapperSingle(Repr):
    """A function wrapper that returns a partial for input only."""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray): return self.function(inp)

class FunctionWrapperDouble(Repr):
    """A function wrapper that returns a partial for an input-target pair."""

    def __init__(self, function: Callable, input: bool = True, target: bool = False, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.input = input
        self.target = target

    def __call__(self, inp: np.ndarray, tar: dict):
        if self.input: inp = self.function(inp)
        if self.target: tar = self.function(tar)
        return inp, tar

class Compose:
    """Baseclass - composes several transforms together."""

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __repr__(self): return str([transform for transform in self.transforms])

class ComposeDouble(Compose):
    """Composes transforms for input-target pairs."""

    def __call__(self, inp: np.ndarray, target: dict):
        for t in self.transforms:
            inp, target = t(inp, target)
        return inp, target

class ComposeSingle(Compose):
    """Composes transforms for input only."""

    def __call__(self, inp: np.ndarray):
        for t in self.transforms:
            inp = t(inp)
        return inp

class AlbuSeg2d(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (C, spatial_dims)
    Expected target: (spatial_dims) -> No (C)hannel dimension
    """
    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target
        out_dict = self.albumentation(image=inp, mask=tar)
        input_out = out_dict['image']
        target_out = out_dict['mask']

        return input_out, target_out

class AlbuSeg3d(Repr):
    """
    Wrapper for albumentations' segmentation-compatible 2D augmentations.
    Wraps an augmentation so it can be used within the provided transform pipeline.
    See https://github.com/albu/albumentations for more information.
    Expected input: (spatial_dims)  -> No (C)hannel dimension
    Expected target: (spatial_dims) -> No (C)hannel dimension
    Iterates over the slices of a input-target pair stack and performs the same albumentation function.
    """

    def __init__(self, albumentation: Callable):
        self.albumentation = A.ReplayCompose([albumentation])

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        # input, target (target has to be in uint8)
        tar = tar.astype(np.uint8)  

        input_copy = np.copy(inp)
        target_copy = np.copy(tar)

        replay_dict = self.albumentation(image=inp[0])['replay']  # perform an albu on one slice and access the replay dict

        # only if input_shape == target_shape
        for index, (input_slice, target_slice) in enumerate(zip(inp, tar)):
            result = A.ReplayCompose.replay(replay_dict, image=input_slice, mask=target_slice)
            input_copy[index] = result['image']
            target_copy[index] = result['mask']

        return input_copy, target_copy