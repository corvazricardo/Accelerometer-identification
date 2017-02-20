
# coding: utf-8

# In[6]:

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix
from matplotlib import cm as cmap


# In[7]:

class Visualization(object):
    
    def __init(self):
        """
        * Generate Visualization object, which creates
        * diverse visualizations of a given set of information
        *
        *
        """
        pass
    
    def isdata_balanced(self,dataset):
        """
        * Plot distribution of classes in data
        """
        dataset.y_label.Id.value_counts().plot(kind='bar',
                                          title="Classes distribution",
                                          figsize=(15,5),grid=True)
        plt.xlabel("Class ID")
        plt.ylabel("No. of instances")
        
        return
    
    def plot_xyz(self,df):
        """
        * Plot time series data of x,y and z 
        """
        
        fig = plt.figure(figsize=(15, 7)) 
        gs = gridspec.GridSpec(3, 1,height_ratios=[1,1,1],hspace=0.7)
        

        ax0 = plt.subplot(gs[0])
        ax0.set_ylabel("Acceleration")
        ax0.set_xlabel("Instance")
        df.x.plot( ax=ax0,title='X Acceleration',grid=True,color="green")
      
        ax1 = plt.subplot(gs[1])
        ax1.set_ylabel("Acceleration")
        ax1.set_xlabel("Instance")
        df.y.plot( ax=ax1,title="Y Acceleration",grid=True,color="orange")
        
        ax2 = plt.subplot(gs[2])
        ax2.set_ylabel("Acceleration")
        ax2.set_xlabel("Instance")
        df.z.plot( ax=ax2,title="Z Acceleration",grid=True)
 
        plt.show()
            
        return
    
    def counter(self,y_data,message):
        """
        * Returns shape of y_data in a visual form 
        """
        self.plot_distribution(y_data,message)
        print(message + ' shape {}'.format(Counter(y_data)))
        
        return
    
    def plot_distribution(self,y_data,message):
        """
        * Plot distribution of data
        """
        
        df_y_train = pd.DataFrame(y_data,columns=['Label'])
        df_y_train.Label.value_counts().plot(kind='bar',grid=True,title= "Distribution of " + message,
                                            figsize=(15,5))
        plt.xlabel('Class ID')
        plt.ylabel('No. of instances')
        plt.show()
        
        return
        
    def plot_confusion_matrix(self,y_test,test_preds,title):
        """
        * Plot confusion matrix of y_test vs test_preds data
        """
        
        cm = confusion_matrix(y_test, test_preds)

        fig, axes = plt.subplots(figsize=(15,6))

        colorbar = axes.matshow(cm, cmap=cmap.gist_heat_r)
        fig.colorbar(colorbar)

        axes.set_xlabel('Predicted class', fontsize=11)
        axes.set_ylabel('True class', fontsize=11)

        plt.title(title)
        plt.show()
        
        return


# In[ ]:



