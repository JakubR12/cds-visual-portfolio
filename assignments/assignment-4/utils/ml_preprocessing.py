# !/isr/bin/python 

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

'''

Module for loading, scaling, and splitting the MNIST784 dataset. T  

'''

def scale(data):
    '''
    min -max scales a numpy array
    
    input:
        data - numpy array  
    '''
    
    output = (data-data.min())/(data.max() - data.min())
    return output

def fetch_visual_data(data_name = "mnist_784", version=1, train_size = 0.8, test_size = 0.2,random_state = None):
    
    '''
    
    fetches, splits and scales datasets using sklearn
    
    inputs:
        data_name: str, default = mnists_784",
        version: int or "active" - default = 1
        train_size: float or int, default = 0.8
        test_size: float or int, default = 0.2
        
    output: 
        X_train, X_test, y_train, y_test: predictors and labels for the split data
        
    '''

    X, y = fetch_openml(name = data_name, version=version, return_X_y=True)

    X = np.array(X)
    y = np.array(y)


    # split data
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                      train_size=train_size,
                                                      test_size=test_size,
                                                       random_state = random_state)

    X_train = scale(X_train)
    X_test = scale(X_test)
    
    return  X_train, X_test, y_train, y_test

if __name__ =="__main__":
    pass
