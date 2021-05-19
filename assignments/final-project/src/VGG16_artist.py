#!/usr/bin/env python

"""
This script uses a pre-trained model called VGG16 as a feature extractor and trains simple neural network classifier on the impressionist paintings.
"""

# basic modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.ticker import MaxNLocator
import os 
import argparse

# sklearn tools
from sklearn.metrics import classification_report

# tf tools
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, 
                                     MaxPooling2D, 
                                     Activation, 
                                     Flatten, 
                                     Dense,
                                     Dropout, #layer to to kill some nodes at random. Sometimes helps with overfitting...
                                     BatchNormalization)  #layer to normalize weight-values. Sometimes helps with performance and overfitting...

from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam # optimizer related to Stochastic gradient descent. Makes the model fit a little faster...
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping #makes the model stop training, when it stops improving
from tensorflow.keras.preprocessing.image import ImageDataGenerator #helps with loading data and generating more training data. https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
from tensorflow.keras.models import Model

# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)



def plot_history(H, epochs,file_name):
    """"
    Function for plotting keras model training history
    """
    plt.style.use("fivethirtyeight")
    fig, ax = plt.subplots()
    ax.plot(H.history["loss"], label="train_loss")
    ax.plot(H.history["val_loss"], label="val_loss")
    ax.plot(H.history["accuracy"], label="train_acc")
    ax.plot(H.history["val_accuracy"], label="val_acc")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Training Loss and Accuracy")
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Loss/Accuracy")
    ax.legend()
    plt.tight_layout()
    plt.draw()
    plt.savefig(file_name)
  

def main(
    model_name ="VVG16_artistic_classifier",
    epochs=50, 
    patience=3,
    dropout=0.3,
    image_size = 128):
    

    '''
        This script uses a pre-trained model called VGG16 as a feature extractor and trains simple neural network classifier on the impressionist paintings.
  
 
     Input:
    model_name, default: VVG16_artistic_classifier,   description: str,   a name of the model and its corresponding plots and weights,
    epochs:     default: 50,                          description: int,   a number of epochs
    patience:   default: 3,                           description: int,   a number of how many times can a model perform worse than in the previous epoch before it is shut down
    dropout:    default: 0.3,                         description: float, a dropout in the network
    image_size: default: 128,                         description: int,   a number of pixels of the images

  Output:
    In to the "models" folder, following things are saved:
        - report metric
        - plot history
        - model weights
        - model architecture
    '''

    # random seeds
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # an if statement to distinguish between binary and categorical classification to adjust necessary parameters within the data pre-processing, modelling and reporting
    with open('artists.txt') as f:
        classes = [line.rstrip() for line in f]

    if len(classes) > 2:
        class_type = "categorical"
        active_func = "softmax"
        end_nodes = len(classes)

    elif len(classes) == 2:
        class_type = "binary"
        active_func = "sigmoid"
        end_nodes = 1
        axis = 1
    else:
        raise SystemExit('You need at least to clases to use the classifier')

    
    # define model
    # load model without classifier layers
    model = VGG16(include_top=False, 
              pooling='avg',
              input_shape=(image_size, image_size, 3))

    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False


    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(256, 
                   activation='relu')(flat1)
    drop1 = Dropout(dropout)(class1)
    output = Dense(end_nodes, 
                   activation=active_func)(drop1)

    # define new model
    model = Model(inputs=model.inputs, 
                  outputs=output)

    # Data preprocessing
    # paths for data
    training_dir = os.path.join("..", "data", "raw", "training", "training")
    test_dir = os.path.join("..", "data", "raw", "validation", "validation") 

    # for a more detail explanation: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    # formula for generating new train data 
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range = 0.3, horizontal_flip = True)

    # rescaling the test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 8

    # streams and resizes data from directory
    train_generator = train_datagen.flow_from_directory(
            training_dir,  # this is the target directory
            target_size=(image_size, image_size),  # all images will be resized to a chosen number
            batch_size=batch_size,
            class_mode=class_type, 
            interpolation = "lanczos", #method for resizing picture that is supposed to yield the highest accuracy
            classes = classes,
            seed = seed,)  

    # streams and resizes data from directory
    validation_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode=class_type,
            interpolation = "lanczos",
            seed = seed,
            classes = classes,
            shuffle = False
            ) #For some reason you shouldn'´t shuffle the validation data. Otherwise you will get chance accuracy when predicting. 

    # argument for early stopping when the model stops improving and save the epoch with the best fit
    callback = EarlyStopping(monitor="val_loss", patience=patience, min_delta = 0.001, restore_best_weights=True)
    
    # model's compilers
    model.compile(loss=f"{class_type}_crossentropy",
                  optimizer= "rmsprop",
                  metrics=["accuracy"])
    

    # print and save model's archite
    achitecture_path = os.path.join("..","models",f"{model_name}_architecture.png")
    plot_model(model, to_file = achitecture_path, show_shapes=True, show_layer_names=True)
    print(model.summary())

    print(f"Image of model architecture saved in {achitecture_path}")
    

    print(f"Fitting a {class_type} classifier with {end_nodes} classes")
    print(f"Classes are: {classes}")
    

    batch_size = 32

    # fit model and save fitting hisotry
    H = model.fit(
            train_generator, #use the resized train data
            epochs=epochs,
            batch_size = batch_size,
             validation_data=validation_generator, #use the resized test data
            verbose=1, #print progress
            callbacks=[callback]
            )
            
    # save and print the weights
    weights_path = os.path.join("..","models", "weights",f"{model_name}_model.h5")
    model.save_weights(weights_path)
    
    print(f"Model's weights saved in {weights_path}")
    
    # plot and save the training history
    history_path = os.path.join("..","models",f"{model_name}_history_plot.png")
    plot_history(H,callback.stopped_epoch,history_path) 
    
    print(f"Training history plot saved in {history_path}")


    # get data labels
    test_labels=validation_generator.classes 
    
    # predict the probability distribution of the data
    predictions=model.predict(validation_generator, verbose=1)


    if class_type == "binary":
        y_pred= predictions.reshape(-1)
        y_pred[y_pred <= 0.5] = 0
        y_pred[y_pred > 0.5] = 1
    
    else:
        # get the class with highest probability for each sample
        y_pred = np.argmax(predictions, axis=-1)


    # get the classification report
    cr = classification_report(test_labels,y_pred, target_names =  validation_generator.class_indices.keys())
    filepath = os.path.join("..","models",f"{model_name}_metrics.txt")
    text_file = open(filepath, "w")
    text_file.write(cr)
    text_file.close()

    print(cr)
    print(f"model metrics saved in {filepath}")
    
    print("Script complete :-)")

if __name__ =="__main__":
    
    # We argparse to add possible inputs from terminal
    ap = argparse.ArgumentParser(description = "[INFO] Calculating network metrics and plotting the network history graph")
    
    ap.add_argument("-n", "--model_name", default = "VVG16_artistic_classifier",
                    type = str, help = "str, a name of the model and its corresponding plots and weights")
        
    ap.add_argument("-e", "--epochs",default = 50,
                    type = int, help = "int, a number of epochs")
    
    ap.add_argument("-p", "--patience", default = 3, type = int,
                    help = "int, a number of how many times can a model perform worse than in the previous epoch before it is shut down")
    
    ap.add_argument("-d", "--dropout", default = 0.3 , type = float, help = "float, a dropout in the network")
    
    ap.add_argument("-is", "--image_size", type = int, default = 128, help = "int, a number of pixels of the images")
    
    
    args = vars(ap.parse_args())

    main(
      args["model_name"],
      args["epochs"],
      args["patience"],
      args["dropout"],
      args["image_size"])
