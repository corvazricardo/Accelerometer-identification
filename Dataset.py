
# coding: utf-8

# In[1]:

import pickle
import pandas as pd   


# In[2]:

class Dataset(object):
    
    def __init__(self ):
        
        self.raw_Xtrain = []
        self.raw_Ytrain = []
        self.raw_Xtest = [] 
        self.y_label = None
        self.ts_train = None
        self.ts_test = None
        
        
        pass
    
    def main(self):
        """
        * The main method to build a Dataset object
        """
        
        self.load_rawdata()
        return
    
    def load_rawdata(self):
        """
        * Load raw data into the Dataset object
        """
        
        self.get_trainTest_data("train")
        self.get_trainTest_data("test")
        
        return
    
    def get_trainTest_data(self,partitionName):
        """
        * Get partitionName raw data
        """
        
        if partitionName == "train":
            self.raw_Xtrain, self.raw_Ytrain = pickle.load(open("dataset/train_X_y.p","rb"))
            
        elif partitionName == "test":
            self.raw_Xtest =  pickle.load(open("dataset/test_X.p","rb"))
        
        return
    
    def mapto_timeseries(self,sect):
        """
        * Map raw data into time series
        """
    
        if sect is "train": 
            self.ts_train = self.get_ts_data(self.raw_Xtrain) 
            self.y_label = pd.DataFrame(self.raw_Ytrain,columns=['Id'])
        
        elif sect is "test":
            self.ts_test = self.get_ts_data(self.raw_Xtest)
            
        return
    
    
    def get_ts_data(self,raw_features):
        """
        * Returns time series data of raw features
        """
    
        ts_data = []

        for fragment in raw_features:
            ts_data.append(pd.DataFrame(fragment,columns=["x","y","z"]))
    
        return ts_data
            
            


# In[ ]:



