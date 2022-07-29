# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:50:20 2022

@author: Vy

This is my practice of Deep NN
Data: Acoustic Fire Extinguisher
Task: Predict the extinguishing status of the fire based on certain levels of sound
and various factors
"""
import tensorflow as tf
#Set up plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex2 import *

import pandas as pd

data= pd.read_csv('A_E_Fire_Dataset.csv')
print(data.head())
print(data.shape)

#The target of this task is 'STATUS'. The remaining columns we'll used as inputs
input_shape= [6]

#Create a model with 3 hidden layers, each having 512 units and the ReLU activation.
# include an output layer of one unit and no activation 
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    #the hidden 3 ReLU layers
    layers.Dense(units=512, activation='relu', input_shape=input_shape),
    layers.Dense(units=512, activation='relu'),
    layers.Dense(units=512, activation='relu'),
    #output layer
    layers.Dense(1)
    ])
q_2.solution()

