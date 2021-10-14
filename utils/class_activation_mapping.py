'''
This module implements the ideas of the paper https://arxiv.org/abs/1512.04150 which is from researches of MIT. 
The basic idea is, that a CNN learns activation maps as types of bounding boxes for free (without the need to label all bounding boxes by hand). 

Steps to get the Class Activation Map (CAM)
(Prerequesite: CNN with Global Average Pooling layer after the last convolutional layer and NO additional dense layers 
(only the output dense layer for getting final predictions!)
    1. Feed image to convolutional network and create perdiction
    2. Fetch the weights connected to the winning neuron
    3. Store the outputs from the last convolutional layer
    4. Use the fetched weights to weight the corresponding output from last convolutional layer and add all values to one final map
    5. Expand this final map (i.e. using bilinear upsampling) to the size of the input image and plot the results
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import scipy 
import matplotlib.pyplot as plt
from PIL import Image

def get_class_activation_map(model, img, name_final_conv_layer):
    ''' 
    this function computes the class activation map
    
    Args:
        model (tensorflow model): trained model
        img (numpy array of shape (height, width, depth)): input image
        name_final_conv_layer (string): The name of the final convolution layer, 
                                        which is the layer before the values go to the global average pooling layer

    Returns:
        final_output (numpy array of same shape as image): Computed class activation map
        label_index (int): The index of the winning neuron that can then be used to get the according label name
    '''
    # get shape of input image
    img_height, img_width, img_depth = img.shape
    
    # expand dimension to fit the image to a network accepted input size
    img = np.expand_dims(img, axis=0)

    # predict to get the winning class
    predictions = model.predict(img)
    label_index = np.argmax(predictions)

    # Get the input weights to the softmax of the winning class.
    class_weights = model.layers[-1].get_weights()[0]
    class_weights_winner = class_weights[:, label_index]
    
    # get the final conv layer
    final_conv_layer = model.get_layer(name_final_conv_layer)
    
    # create a function to fetch the final conv layer output maps 
    get_output = K.function([model.layers[0].input],[final_conv_layer.output, model.layers[-1].output])
    [conv_outputs, predictions] = get_output([img])
    
    # squeeze conv map
    conv_outputs = np.squeeze(conv_outputs)
    
    # bilinear upsampling to resize each filtered image to size of original image 
    fac_height = img_height / conv_outputs.shape[0]
    fac_width = img_width / conv_outputs.shape[1]
    num_channels = conv_outputs.shape[-1]
    mat_for_mult = scipy.ndimage.zoom(conv_outputs, (fac_height, fac_width, 1), order=1)

    # get class activation map for object class that is predicted to be in the image
    final_output = np.dot(mat_for_mult.reshape((img_width*img_height, num_channels)), class_weights_winner).reshape(img_height, img_width)
    
    # return class activation map
    return final_output, label_index


def plot_class_activation_map(CAM, img, label, figsize=(8, 8)):
    ''' 
    this function plots the activation map 
    
    Args:
        CAM (numpy array of shape (img_height, img_width)) : class activation map containing the trained heat map
        img (numpy array of shape (img_height, img_width, img_depth)) : input image
        label (string) : label of the winning class (used as title)
        figsize (tuple of floats): Figure width and height
    '''
    
    fig = plt.figure(figsize=figsize)

    # plot image
    plt.imshow(img, alpha=0.5, cmap='gray')
    
    # plot class activation map
    plt.imshow(CAM, cmap='jet', alpha=0.5)
    
    # get string for classified class
    plt.title(label)

    plt.show()

def compute_and_plot_CAM(model, img, labels_list, name_final_conv_layer, input_rgb=True, figsize=(8, 8)):
    '''
    This function takes a trained tensorflow model and an image and plots the resulting class activation map.

    Args:
        model (tensorflow model): trained model
        img (numpy array of shape (height, width, depth)): input image
        labels_list (list of strings): List of labels or None if label should not be plotted
        name_final_conv_layer (string): The name of the final convolution layer, 
                                        which is the layer before the values go to the global average pooling layer
        input_rgb (boolean): Flag to indicate whether the input image is an RGB image -> should be converted to grayscale for plotting the CAM                                        
        figsize (tuple of floats): Figure width and height
        '''

    # get CAM
    cam, label_index = get_class_activation_map(model, img, name_final_conv_layer)

    # get label if required
    if labels_list != None:
        label = labels_list[label_index]
    else:
        label = "None"

    # convert image to grayscale if it is RGB
    if(input_rgb):
        img = np.asarray(Image.fromarray((img  * 255).astype(np.uint8)).convert("L"))
    plot_class_activation_map(cam, img, label, figsize=figsize)