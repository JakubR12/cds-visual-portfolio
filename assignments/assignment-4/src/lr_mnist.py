#!/usr/bin/python

# importin modules
import argparse
import os
import sys
sys.path.append(os.path.join(".."))
import numpy as np

# Import utils for classification, preprocessing
import utils.classifier_utils as clf_util
import utils.ml_preprocessing as mlp

# Import sklearn metrics
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

'''


There are 7 arguments which can but do not have to be specified:

flags: -tr, --train_size,  default: 0.8:  description: int or float, a proportion of the data the model to be trained on
flags: -ts, --test_size:  default: 0.2, description: int or float, a proportion of the data the model to be trained on
flags: -r, --random_state: default: None, description:   int, a random seed
flags: -sm, --save_metrics: default: None, description: bool, whether to save the metrics
flags: -mf, --metrics_safe, default: logistic_regression_metrics.txt, description: str, the filename of the metrics with the .txt ending
flags: -t, --tolerance, default: 0.1, description: float, Tolerance for stopping criteria
flags: -p, --penalty, default: None, description: "none" "l1", ‘l2’, "elasticnet

examples:
  python lr-mnist.py -tr 0.7 -ts 0.3 -r 2 -sm -mf log_reg_filename.txt -t 0.001 -p l2 
  
  When using boolean flags (-sm), just leave it empty.

'''

def main(train_size = 0.8,
         test_size = 0.2, 
         random_state = None, 
         save_metrics = True, 
         metrics_filename = "logistic_regression_metrics.txt",
         penalty= "none",
         tolerance = 0.1):
    
    
    """
    
    This functions trains a logistic regression classifier on the mnist data set and prints the metrics.
    
    input:
    
      train_size,  default: 0.8:  description: int or float, a proportion of the data the model to be trained on
      test_size:  default: 0.2, description: int or float, a proportion of the data the model to be trained on
      random_state: default: None, description:   int, a random seed
      save_metrics: default: True, description: bool, whether to save the metrics
      metrics_filename, default: logistic_regression_metrics.txt, description: str, the filename of the metrics with the .txt ending
      tolerance, default: 0.1, description: float, Tolerance for stopping criteria
      penalty, default: None, description: "none" "l1", ‘l2’, "elasticnet
    
    output:
      classification report as a string. Can optionally be saved as a txt file
    
    """
    print("Preparing data. This may take a while ...")
    
    # using a function we built in the ml_preprocessing.py script to load, format, split and scale the data -> we get a clean data
    X_train, X_test, y_train, y_test = mlp.fetch_visual_data(train_size = train_size, test_size = test_size, random_state = random_state)
    
    print("The MNIST dataset has been loaded, split and scaled.")
    
    # calling a logistic regression with customizable arguments
    clf = LogisticRegression(penalty='none', 
                         tol=0.1, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train, y_train)
    
    
    # calculating predictions for the test data, printing output metrics
    y_pred = clf.predict(X_test)
    cm = metrics.classification_report(y_test, y_pred)
    print(cm)
    
    # optional argument to save the data in the out file as a text file
    if save_metrics == True: 
        filepath = os.path.join("..","models",metrics_filename)
        text_file = open(filepath, "w")
        text_file.write(cm)
        text_file.close()
        print("The metric was saved into models folder")

    
if __name__=="__main__":
    
    # define comman line interface arguments
    ap = argparse.ArgumentParser()
    
    ap.add_argument("-tr", "--train_size", required = False, default = 0.8, 
                    help = "int or float: A proportion of the data the model to be trained on")
    ap.add_argument("-ts", "--test_size", required = False, default = 0.2, 
                    help = "int or float:  A proportion of the data the model to be tested on")
    ap.add_argument("-r", "--random_state", required = False, default = None, type = int, 
                    help = "int: a random seed")
    ap.add_argument("-sm","--save_metrics", required = False, action = "store_true", 
                    help = "bool: whether to save the metrics")
    ap.add_argument("-mf", "--metrics_filename", required = False, default =  "logistic_regression_metrics.txt",type = str, 
                    help = "the filename of the metrics with the .txt ending")
    ap.add_argument("-t", "--tolerance", required = False, default = 0.1, type = float, 
                    help = "float: Tolerance for stopping criteria")
    ap.add_argument("-p", "--penalty", required = False, default = "none", type = str, 
                    help = '''penalty: "none" "l1", ‘l2’, "elasticnet"''')
    
    # parse arguments, parse the argumets and returns them as a list of arguments/variables
    args = vars(ap.parse_args())
    print(args)
    
   # instead of listing all comman line interface arguments separately, we can list all of them at once with **args
    main(**args)
