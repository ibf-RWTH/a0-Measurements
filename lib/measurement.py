import cv2
import math
import numpy as np
from skimage.measure import label, regionprops

#%%
class Macroscale():
    """Class to perform initial crack size measurements on prediction masks of macroscale fracture surface images"""
    def __init__ (self,
                  inputs: list,
                  label_mask,
                  resize_dim,
                  nominal_specimen_width,
                  nominal_specimen_thickness,
                  specimen_type
                  ):
              
        self.inputs = inputs
        self.label_mask = label_mask
        self.resize_dim = resize_dim
        self.nominal_specimen_width = nominal_specimen_width
        self.nominal_specimen_thickness = nominal_specimen_thickness
        self.specimen_type = specimen_type
        
        print(self.inputs)
        # resize image according to resize_dim
        self.resized_label_mask = cv2.resize(np.array(self.label_mask), self.resize_dim, interpolation = cv2.INTER_NEAREST)
        # print(np.unique(self.resized_label_mask, return_counts=True))
        self.regions = regionprops(self.resized_label_mask)
        print(f'Mask shape: {self.resized_label_mask.shape}')
        print(f'Number of classes in image: {len(self.regions)}')
        
        has_SG = self._check_for_class(self.resized_label_mask, 1)
        has_EN = self._check_for_class(self.resized_label_mask, 2)
        has_PC = self._check_for_class(self.resized_label_mask, 3)
        has_DF = self._check_for_class(self.resized_label_mask, 4)
        has_BF = self._check_for_class(self.resized_label_mask, 5)
        has_OT = self._check_for_class(self.resized_label_mask, 6)
                
        if (self.specimen_type == 'SE(B)' or self.specimen_type == 'Other with side groove.'):
            # region 0 (background is ignored) --> [0]: side groove, [1]: erosion notch etc.
            if has_SG:
                self.sg_region = self.regions[0]
            if has_EN:
                self.en_region = self.regions[1]
            if has_PC:
                self.pc_region = self.regions[2]
            if has_DF:
                self.df_region = self.regions[3]
                if has_BF:
                    self.bf_region = self.regions[4]
                if has_OT:
                    self.ot_region = self.regions[5]
            else:
                self.bf_region = self.regions[3]
                if has_OT:
                    self.ot_region = self.regions[4]
        elif (self.specimen_type == 'Chevron' or self.specimen_type == 'C(T)' or self.specimen_type == 'Other without side groove.'):
            # region 0 (background is ignored) --> [0]: erosion notch, [1]: erosion notch etc.
            if has_SG:
                self.sg_region = self.regions[0]
                if has_EN:
                    self.en_region = self.regions[1]
                if has_PC:
                    self.pc_region = self.regions[2]
                if has_DF:
                    self.df_region = self.regions[3]
                    if has_BF:
                        self.bf_region = self.regions[4]
                    if has_OT:
                        self.ot_region = self.regions[5]
            else:
                self.en_region = self.regions[0]
                if has_PC:
                    self.pc_region = self.regions[1]
                else:
                    self.pc_region = 0
                if has_DF:
                    ###
                    if len(self.regions)==2:
                        self.df_region = self.regions[1]
                    else:
                        self.df_region = self.regions[2]
                        if has_BF:
                            self.bf_region = self.regions[3]
                        if has_OT:
                            self.ot_region = self.regions[4]
                            ###
                else:
                    self.bf_region = self.regions[3]
                    if has_OT:
                        self.ot_region = self.regions[4]
        elif self.specimen_type == 'Other':
            something=2 
        
        self.specimen_mm_area = self.nominal_specimen_thickness * self.nominal_specimen_width
        
        opt = self.resized_label_mask
        skbinary_mask = self._largest_region_binary(self.resized_label_mask)
        _, counts = np.unique(skbinary_mask, return_counts=True)
        self.specimen_pixel_area = counts[1]
        
        thicknessandwidth = self._measure_thicknessandwidth(opt, 64)
        self.pixel_thickness_B = thicknessandwidth[1]
        self.pixel_width_W = thicknessandwidth[0]
        
        # If there are no side grooves B = BN
        if (self.specimen_type == 'SE(B)' or self.specimen_type == 'Other with side groove.'):
            self.pixel_netthickness_BN = self._measure_netthickness(prediction=opt, pc_region=self.pc_region, en_region=self.en_region)
        else:
            self.pixel_netthickness_BN = self.pixel_thickness_B
        
        self.pixel_area_erosionnotch = self._slice_erosionnotch(prediction=opt, en_region=self.en_region, pixel_netthickness_BN=self.pixel_netthickness_BN)
        self.pixel_starternotchlength_ak = self.pixel_area_erosionnotch / self.pixel_netthickness_BN
        self.specimen_orientation = self.regions[0].orientation
        self.scale = math.sqrt(self.specimen_mm_area / self.specimen_pixel_area) #area
        self.thickness_B = self.pixel_thickness_B * self.scale
        self.width_W = self.pixel_width_W * self.scale
        self.netthickness_BN = self.pixel_netthickness_BN * self.scale
        self.starternotchlength_ak = self.pixel_starternotchlength_ak * self.scale
    
    def get_area_average(self):
        return self._area_average()
    
    def get_dimensions_and_scale(self):
        return self._dimensions_and_scale()
    
    def get_thicknessandwidth(self):
        return self._measure_thicknessandwidth()
    
    def get_netthickness(self):
        return self._measure_netthickness()
    
    def _return_resizedimg(self):
        return self.resized_label_mask
    
    def _return_bboxes(self):
        return self.spec_bbox, self.pre_bbox, self.ero_bbox
    
    def _dimensions_and_scale(self):
        return self.specimen_orientation, self.scale, self.thickness_B, self.netthickness_BN, self.width_W, self.starternotchlength_ak
        
    def _area_average(self):
        """
        Implementation of the area average method. The initial crack size is calculated as the quotient of crack area (erosion notch + fatigue precrack) devided by
        the net thickness BN

        Returns
        -------
        pixel_a0 : float
            Initial crack size in px.
        initial_crack_size_a0 : float
            Initial crack size in mm.
        crack_aspect_ratio : float
            Value of a0/W.

        """
        ero_area = self.pixel_area_erosionnotch
        if self.pc_region == 0:
            pre_crack_area = 0
        else:
            pre_crack_area = self.pc_region.area
        
        #area average method
        pixel_a0 = (ero_area + pre_crack_area) / self.pixel_netthickness_BN
        
        if 6 in np.unique(self.resized_label_mask):
            gauge_notch_width = self.regions[len(self.regions)-1].area / self.pixel_thickness_B
            pixel_a0 = pixel_a0 + gauge_notch_width
        
        initial_crack_size_a0 = pixel_a0 * self.scale
        
        crack_aspect_ratio = initial_crack_size_a0 / self.width_W
          
        return pixel_a0, initial_crack_size_a0, crack_aspect_ratio

    def _check_for_class(self, prediction, value):
        """
        Checks wether label is in prediction mask.

        Parameters
        ----------
        prediction : np.array
            Prediction mask.
        value : int
            Class label (value between 1 and 6).

        Returns
        -------
        bool
            True if label is in prediction mask.

        """
        return np.any(prediction == value)
    
    def _measure_thicknessandwidth(self, prediction, num_sectors):
        """
        Thickness B and width W are measured in pixels by counting the non zero values per row and column in the prediction mask for a predefined number of sectors. 
        At least a 10th of the images must show the specimen in order for the algorithm to work.

        Parameters
        ----------
        prediction : np.array
            Prediction mask.
        num_sectors : int
            Number of measurements.

        Returns
        -------
        width_W : float
            Width in px.
        thickness_B : float
            Thickness in px.

        """
        width_sum = 0
        thickness_sum = 0
        width_measures = 0
        thickness_measures = 0
        
        xpred_max = prediction.shape[1]
        ypred_max = prediction.shape[0]
        x_step = xpred_max / num_sectors
        y_step = ypred_max / num_sectors
        
        for sector in range(num_sectors-1):
            col = int((sector+1)*x_step)
            
            # If there is only background or a small misclassification, the row shall not be considered
            if np.count_nonzero(prediction[:, col]) <= ypred_max/10: 
                continue
            else:
                width_count = np.count_nonzero((prediction[:, col] != 0))
                width_sum += width_count
                width_measures += 1
                
        for sector in range(num_sectors-1):
            row = int((sector+1)*y_step)    
            # If OTHER class is present in row, do not measure the thickness in this row.
            if np.count_nonzero(prediction[row]) <= xpred_max/10: 
                continue
            else:
                thickness_count = np.count_nonzero((prediction[row] != 0))
                thickness_sum += thickness_count
                thickness_measures += 1
            
        width_W = width_sum / width_measures
        thickness_B = thickness_sum / thickness_measures
       
        print(f'Number of width / thickness measurements: {width_measures} / {thickness_measures}')
            
        return width_W, thickness_B
    
    def _measure_netthickness(self, prediction, pc_region, en_region):
        """
        Calculates the net thickness BN based on a bounding box around the erosion notch and fatigue precrack.

        Parameters
        ----------
        prediction : np.array
            Prediction mask.
        pc_region : skimage.measure.regionprops region
            Fatigue precrack region.
        en_region : skimage.measure.regionprops region
            Erosion notch region.

        Returns
        -------
        pixel_netthickness_BN : float
            Net thickness in px.

        """
        # Outer boundaries of erosion notch
        en_bbox = en_region.bbox
        etop, eleft, ebottom, eright = en_bbox 
        # Outer boundaries of precrack
        pre_bbox = pc_region.bbox
        ptop, pleft, pbottom, pright = pre_bbox 
        # Centroid
        centroid_row = prediction[int(pc_region.centroid[0])]
        centroid_netthickness_BN = np.count_nonzero(centroid_row==3)
        
        bbox_netthickness_BN = min(pright - pleft, eright - eleft)
        pixel_netthickness_BN = (centroid_netthickness_BN + bbox_netthickness_BN) / 2
        
        return pixel_netthickness_BN
    
    def _slice_erosionnotch(self, prediction, en_region, pixel_netthickness_BN):
        """
        Returns the area of the erosion notch.

        Parameters
        ----------
        prediction : np.array
            Prediction mask.
        en_region : skimage.measure.regionprops region
            Erosion notch region.
        pixel_netthickness_BN : float
            Net thickness in px.

        Returns
        -------
        area_erosionnotch_afterslicing : float
            Area of the erosion notch in px.

        """
        rows_with_en = np.where(np.any(prediction == 2, axis=1))[0]
        
        # From centroid of the erosion notch slice away everything except the netthickness
        centroid_col = int(en_region.centroid[1])
        predthickness = prediction.shape[1]
        left_slice = int(centroid_col - pixel_netthickness_BN/2)
        right_slice = int(centroid_col + pixel_netthickness_BN/2)
        sliced_array = prediction[:, left_slice:right_slice]

        # Count the number of erosion notch pixels after slicing
        area_erosionnotch_afterslicing = np.count_nonzero(sliced_array == 2)
    
        return area_erosionnotch_afterslicing
    
    def _largest_region_binary(self, resized_label_mask):
        """
        Binarizes the image: 0 - background, 1 - specimen.

        Parameters
        ----------
        resized_label_mask : np.array
            Resized prediction mask.

        Returns
        -------
        largest_region_mask : np.array
            Binary prediction mask.

        """
        binary_mask = (np.where(resized_label_mask > 0, 1, resized_label_mask)).astype(np.uint8)
                 
        # Create unique labels for connected components (otherwise all 1s would be treated as one region)
        sklabeled_mask = label(binary_mask)
        
        # Calculate region properties to find the largest region
        regions = regionprops(sklabeled_mask)
        
        # Find the largest region based on area
        largest_region = max(regions, key=lambda r: r.area)
        largest_label = largest_region.label
        
        # Create a new mask keeping only the largest region
        largest_region_mask = np.where(sklabeled_mask == largest_label, binary_mask, 0)
        return largest_region_mask
