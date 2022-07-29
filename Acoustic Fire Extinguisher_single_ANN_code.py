# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""
This is my practices of single ANN model. 
Data: Acoustic Fire Extinguisher
"""
#Setup plotting
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

#Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex1 import *

#Run the cell
import pandas as pd
data = pd.read_csv('A_E_Fire_Dataset.csv')
print(data.head())
print(data.shape)

#Choose STATUS column as the target
#Define the linear model
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([layers.Dense(units=1, input_shape=[6,])])


#Check the weights
w, b= model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))
