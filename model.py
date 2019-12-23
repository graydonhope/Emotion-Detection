import cv2
import numpy as np
import os
import math
import time
import tensorflow as tf
import sklearn
import pandas as pd
import ProcessedData
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import array
from sklearn.model_selection import train_test_split, GridSearchCV

def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

def load_image(path):
    image = cv2.imread(path)
    return image

def get_max_dimens(images):
    """ 
    Finds the largest size x and y value dimensions in the loaded images

    Parameters:
    images: The images with different dimensions

    Returns:
    (int, int): the max x and y values of the image lists 
    """
    if images is not None:
        x_max = 0
        y_max = 0
        for i in range(len(images)):
            img = images[i]
            if (img.shape[0] > x_max):
                x_max = img.shape[0]
            if (img.shape[1] > y_max):
                y_max = img.shape[1]
    
        return (x_max, y_max)
    else:
        raise ValueError('images value is None')

def convert_to_grayscale(images):
    for i in range(len(images)):
        img = images[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images[i] = img
    
    return images

def resize_images(images, x, y):
    resized_images = []
    dimension = (y, x)

    for i in range(len(images)):
        img = images[i]
        resized = cv2.resize(img, dimension)
        resized_images.append(resized)
    
    return resized_images

def preprocess_data(X, y):
    # The data will be split into a 0.75, 0.20, and 0.05 split for training, test, and CV respectively.

    # Convert the input list into a numpy array
    X = array(X)
    y = array(y)

    # Split the data into train and test
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Create validation set and scale the pixel values so they are between 0 and 1 (note pixel values
    # ... range from 0 to 255).
    X_validation, X_train = X_train_full[:360]  / 255.0, X_train_full[360:] / 255.0
    y_validation, y_train = y_train_full[:360], y_train_full[360:]

    class_names = ["Not Smiling", "Smiling"]

    data = ProcessedData.ProcessedData(X_train, X_test, X_validation, y_train, y_test, y_validation, class_names)

    return data

def create_model():
    # Create a Sequential multi-layer perceptron network with a sigmoid output activation function for binary classification
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[24, 49]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(50, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    return model

def compile_model(model, loss_function, optimizer):
    # The choice of loss function is directly related to the activation function in the output layer
    # ... of the neural network
    model.compile(loss=loss_function,
                optimizer=optimizer, 
                metrics=["accuracy"])

def visualize_train(model):
    pd.DataFrame(model.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(-0.5, 1.5)
    plt.show()

def tensorboard_visualization():
    root_logdir = os.path.join(os.curdir, "logs")
    
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

def main():
    no_smile_images = load_images(os.path.join(os.curdir, "data\\no_smile_mouth"))
    smile_images = load_images(os.path.join(os.curdir, "data\\mouth"))
    gray_no_smile_images = convert_to_grayscale(no_smile_images)    
    gray_smile_images = convert_to_grayscale(smile_images)  
    max_dimens = get_max_dimens(no_smile_images)
    max_smile_dimens = get_max_dimens(smile_images)

    # Resize all the grayed out images so that the model has consistent input dimensions
    resized_images = resize_images(gray_no_smile_images, max_dimens[0], max_dimens[1])
    resized_smile_images = resize_images(gray_smile_images, max_dimens[0], max_dimens[1])

    # Create the labels
    no_smile_labels = [0] * (len(resized_images))
    smile_labels = [1] * (len(resized_smile_images))

    # Join the lists
    X = resized_images + resized_smile_images
    y = no_smile_labels + smile_labels

    # Split the data into train, test, and CV sets
    preprocessed_data = preprocess_data(X, y)
    
    X_train = preprocessed_data.get_X_train()
    y_train = preprocessed_data.get_y_train()
    X_valid = preprocessed_data.get_X_valid()
    y_valid = preprocessed_data.get_y_valid()
    
    # Create the model 
    model = create_model()

    # Compile the model using sparse-categorical cross entropy loss function because the labels are integers
    compile_model(model, loss_function="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=0.0001))

    # Train the model and visualize with Tensorboard - use command: tensorboard --logdir=logs --port=6006
    run_logdir = tensorboard_visualization()
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    fitted_model = model.fit(X_train, y_train, epochs=50, 
                            validation_data=(X_valid, y_valid),
                            callbacks = [tensorboard_cb])

    # Visualize the training metrics
    visualize_train(fitted_model)

    # Evaluate the trained model on the test set
    X_test = preprocessed_data.get_X_test()
    y_test = preprocessed_data.get_y_test()
    # model.evaluate(X_test, y_test)

    # Make a sample prediction - use first 5 instances of test set since we don't have any new instances
    X_new = X_test[:5]
    y_new = y_test[:5]
    y_proba = model.predict(X_new)
    y_pred = model.predict_classes(X_new)
    class_names = preprocessed_data.get_classnames()
    print(np.array(class_names)[y_pred])
    # Actual values (see if they match the predicted values - they do)
    print(np.array(class_names)[y_new])

    # Save the trained model
    model.save("smile_detection_model.h5")
   
main()