
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
from detect_peaks import detect_peaks


# In[2]:

class Manipulate(object):
    
    def __init__(self,df=None):
        """
        * A preprocessing object which allows performing most of the 
        * required transformations to manipulate dataset
        """
        pass
    
    def build_features(self,dt,sect='train'):
        """
        * Build features of dataset,which will feed a given model
        """
        
        all_data = []
        
        if sect is "train":
            data = dt.ts_train
        elif sect is "test":
            data = dt.ts_test 
        
        if data:

            for tseries in data:
                
                mean_x, std_x, min_x, max_x, median_x, var_x, autcorr_x = self.get_basic_features(tseries,"x")
                mean_y, std_y, min_y, max_y, median_y, var_y, autcorr_y = self.get_basic_features(tseries,"y")
                mean_z, std_z, min_z, max_z, median_z, var_z, autcorr_z = self.get_basic_features(tseries,"z")
                
                c_xy, c_xz, c_yz, kur_x, kur_y, kur_z, skew_x, skew_y, skew_z = self.get_ts_features(tseries)
                
                peaks_x, peaks_y, peaks_z, zcros_x, zcros_y, zcros_z, mcros_x, mcros_y, mcros_z = self.get_deriv_features(tseries)
            
                inicial_x, inicial_y, inicial_z, final_x, final_y, final_z = self.top_tail_values(tseries)
                
                conversion = [mean_x,  std_x,   min_x,   max_x,   median_x,  var_x,     autcorr_x,
                              mean_y,  std_y,   min_y,   max_y,   median_y,  var_y,     autcorr_y, 
                              mean_z,  std_z,   min_z,   max_z,   median_z,  var_z,     autcorr_z,
                              c_xy,    c_xz,    c_yz,    kur_x,   kur_y,     kur_z,     skew_x, 
                              skew_y,  skew_z,  peaks_x, peaks_y, peaks_z,   zcros_x,   zcros_y, 
                              zcros_z, mcros_x, mcros_y, mcros_z, inicial_x, inicial_y, inicial_z,
                              final_x, final_y, final_z
                             ]
                
                all_data.append(conversion)
                
        else:
            raise TypeError("Empty data provided for building features")
        return all_data
    
    
    
    def get_basic_features(self,ts,column_id):
        """
        * Extract basic features (e.g. Mean,Standard Deviation, Min and Max values) from 
        * a fiven time series
        """
        col_data = ts[column_id]
        
        mean_col = col_data.mean()
        std_col = col_data.std()
        min_col = col_data.min()
        max_col = col_data.max()
        median_col = col_data.median()
        var_col = col_data.var()
        autcorr_col = col_data.autocorr()
        
        return mean_col,std_col, min_col, max_col, median_col, var_col, autcorr_col
    
    def get_ts_features(self,ts):
        """
        * Extract time series related features (e.g. kurtosis, skewness) and the cross correlation
        * between other time series (specifically between 3 axis time series: x_axis,y_axis and z_axis)
        """
        
        corr_xy, corr_xz, corr_yz = ts.corr()['x']['y'], ts.corr()['x']['z'], ts.corr()['y']['z'] 
        kurtosis_x, kurtosis_y, kurtosis_z = ts.kurtosis()['x'], ts.kurtosis()['y'], ts.kurtosis()['z']
        skewness_x, skewness_y, skewness_z = ts.skew()['x'], ts.skew()['y'], ts.skew()['z']
        
        
        return corr_xy, corr_xz, corr_yz, kurtosis_x, kurtosis_y, kurtosis_z, skewness_x, skewness_y, skewness_z

     
    def get_deriv_features(self,ts):
        """
        * Return derivative features of a given time series
        """
        
        peaks_x, peaks_y, peaks_z = self.get_peaks(ts)
        zcros_x, zcros_y, zcros_z = self.get_zero_cross(ts)
        mcros_x, mcros_y, mcros_z = self.get_mean_cross(ts)
        
        return peaks_x, peaks_y, peaks_z, zcros_x, zcros_y, zcros_z, mcros_x, mcros_y, mcros_z

        


    def get_peaks(self,ts):
        """
        * Return number of peaks found in each of the three time
        * series (x_axis ,y_axis and z_axis) contained in ts 
        """
        
        peaks_x = len(detect_peaks(ts.x))
        peaks_y = len(detect_peaks(ts.y))
        peaks_z = len(detect_peaks(ts.z))
        
        return peaks_x, peaks_y, peaks_z
    
    def get_zero_cross(self,ts):
        """
        * Return number of times that each of the three time 
        * series (x_axis ,y_axis and z_axis)
        * crossed zero value
        """
        
        zero_crossings = []
        timeseries = ["x","y","z"]
        
        for tserie in timeseries:
            tmp_zero_cross = len(np.where(np.diff(np.sign(ts[tserie])))[0])
            zero_crossings.append(tmp_zero_cross)
        
        return zero_crossings[0], zero_crossings[1], zero_crossings[2]
    
    def get_mean_cross(self,ts):
        """
        * Return number of times that values of each of the three time 
        * series (x_axis ,y_axis and z_axis)
        * crossed mean value 
        """
        mean_crossings = []
        timeseries = ["x","y","z"]
        
        for tserie in timeseries:

            tmp_mean_cross =  len(np.where(np.diff(np.sign(ts[tserie] - ts[tserie].mean())))[0])
            mean_crossings.append(tmp_mean_cross)
        
        return mean_crossings[0], mean_crossings[1], mean_crossings[2]
    
    def top_tail_values(self,ts):
        """
        * Return initial and final values in each axis of
        * the three time series
        """
        inicial_x, inicial_y, inicial_z = ts.x[0], ts.y[0], ts.z[0]
        final_x, final_y, final_z = ts.x.tail(1).values[0], ts.y.tail(1).values[0], ts.z.tail(1).values[0]
        
        return inicial_x, inicial_y, inicial_z, final_x, final_y, final_z


# In[ ]:



