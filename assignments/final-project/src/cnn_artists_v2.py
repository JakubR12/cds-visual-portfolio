#!/usr/bin/env python

"""
This script trains a CNN classifier on impressionist paintings in attempt to classify the painters of the paintings
"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os 

# sklearn tools
from sklearn.metrics import classification_report

# tf tools
#import tensorflow as tf
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

# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)

# generic model object
from tensorflow.keras.models import Model

painters = ["Matisse","Monet","Sargent"]

#paths for data
training_dir = os.path.join("..", "data", "raw", "training", "training")
test_dir = os.path.join("..", "data", "raw", "validation", "validation") 

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
    fig.savefig(file_name)
  

def main(classes):
  
  
    # random seed
    seed = 42
    np.random.seed(seed)

  
    
    tf.random.set_seed(seed)
    
    image_size = 128 #set the height and width of images
    # define model
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
    output = Dense(3, 
                   activation='softmax')(class1)

    # define new model
    model = tensorflow.keras.model.Model(inputs=model.inputs, 
                  outputs=output)
    # summarize
    model.summary()





    # for a more detail explanation: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
    # formula for generating new train data 
    train_datagen = ImageDataGenerator(rescale=1./255, zoom_range = 0.3, vertical_flip = True)


    # rescaling the test data
    test_datagen = ImageDataGenerator(rescale=1./255)

    batch_size = 8

    # streams and resizes data from directory
    train_generator = train_datagen.flow_from_directory(
            training_dir,  # this is the target directory
            target_size=(image_size, image_size),  # all images will be resized to 64 x 64
            batch_size=batch_size,
            class_mode='categorical', #since we use categorical_crossentropy loss, we need categorical
            interpolation = "lanczos",
            seed = seed,
            classes = classes)  #method for resizing picture that is supposed to yield the highest accuracy

    # streams and resizes data from directory
    validation_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(image_size, image_size),
            batch_size=batch_size,
            class_mode='categorical',
            interpolation = "lanczos",
            seed = seed,
            shuffle = False,
            classes = classes) #For some reason you shouldn'´t shuffle the validation data. Otherwise you will get chance accuracy when predicting. 

    # argument for early stopping of the model once it is not improving anymore
    callback = EarlyStopping(monitor="val_loss", patience=3, min_delta = 0.001, restore_best_weights=True)
    
    # the optimizer Adam is an optimization of the stochastic gradient descent (better results in shorter time)
    opt = Adam()
    model.compile(loss="categorical_crossentropy",
                  optimizer= opt,
                  metrics=["accuracy"])
                  
    print(model.summary())
    achitecture_path = os.path.join("..","models","model_architecture.png")
    #plot model
    plot_model(model, to_file = achitecture_path, show_shapes=True, show_layer_names=True)
    print(f"Image of model architecture saved in {achitecture_path}")
    batch_size = 32
    epochs = 15
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
    weights_path = os.path.join("..","models", "weights","model.h5")
    model.save_weights(weights_path)
    
    print(f"Model's weights saved in {weights_path}")
    
    #plot and save the training hisory
    history_path = os.path.join("..","models","history_plot.png")
    print(callback.stopped_epoch)
    plot_history(H,callback.stopped_epoch,history_path) 
    
    print(f"Training history plot saved in {history_path}")

    # get the ground truth of your data. 
    test_labels=validation_generator.classes 
  
    # predict the probability distribution of the data
    predictions=model.predict(validation_generator, verbose=1)

    # get the class with highest probability for each sample
    y_pred = np.argmax(predictions, axis=-1)

    # get the classification report
    cr = classification_report(test_labels, y_pred, target_names =  validation_generator.class_indices.keys())
    print(cr)
    filepath = os.path.join("..","models","metrics.txt")
    text_file = open(filepath, "w")
    text_file.write(cr)
    text_file.close()
    print(f"model metrics saved in {filepath}")
    
    print("Script complete :-)")
if __name__=="__main__":
    main(painters)
