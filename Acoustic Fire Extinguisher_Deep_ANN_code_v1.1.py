# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:50:20 2022

@author: Vy

This is my sample of Deep NN
Data: Acoustic Fire Extinguisher
Task: Predict the extinguishing status of the fire based on certain levels of sound
and various factors
"""

# Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
data= pd.read_csv('A_E_Fire_Dataset.csv')
print(data.head())
#Seperate target from features
X = data.drop(['STATUS'], axis=1)
y = data.STATUS

#Divide data into training and validation subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=5, shuffle=True)

#Since the FUEL column is a catogorical data. Need to define preprocessing data
from sklearn.preprocessing import OneHotEncoder

#Preprocessing for numerical and categorical data
# Get list of categorical variables
s = (X_train.dtypes == 'object') #This give True/False result of object type columns
object_cols = list(s[s].index) #Colect index data of the column with categorical variables

# Apply one-hot encoder to each column with categorical data
OH_model = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_model.fit_transform(X_train[object_cols]))
OH_cols_test = pd.DataFrame(OH_model.transform(X_test[object_cols]))

# One-hot encoding removed index; now we put it back
OH_cols_train.index = X_train.index
OH_cols_test.index = X_test.index

# Get feature name for OH encoder. Without it, feature name is "x0", "x1",.."nfeature" by defaults
OH_cols_train.columns = OH_model.get_feature_names_out()
OH_cols_test.columns = OH_model.get_feature_names_out()

# Remove categorical columns (then replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

#Add one-hot encoding to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_test, OH_cols_test], axis=1)
print(OH_X_train.head())
print(OH_X_train.shape)


#The target of this task is 'STATUS'. The remaining columns we'll used as inputs
input_shape= [9]

#Create a model with 3 hidden layers, each having 512 units and its own ReLU activation.
# include an output layer of one unit and no activation 
from tensorflow import keras
from tensorflow.keras import layers, callbacks

#First we set callbacks
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, #minimum amount of change to count as improvement
    patience=20, #How many epochs to wait before stopping
    restore_best_weights=True,
    )

model = keras.Sequential([
    #the hidden 3 ReLU layers
    layers.Dense(units=400, activation= 'relu', input_shape=input_shape, name="layer_1"),
    layers.Dropout(0.3), #Apply 30% drop out to the next layers. Making it much harder for the network to learn those spurious patterns in the training data.
    layers.BatchNormalization(), #Correct training data that's slow or unstable
    layers.Dense(units=400, activation= 'relu', name="layer_2"),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(units=400, activation= 'relu', name="layer_3"),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    #output layer
    layers.Dense(1, name="layer_4")
    ])
#Compile the optimizer and loss function
            #Define A "loss function" that measures how good the network's predictions are.
model.compile(optimizer='sgd',
              #Define An "optimizer" that can tell the network how to change its weights.
              loss='mae',
              metrics=['mae'],
              )
#Next, we train the model
history = model.fit(
    OH_X_train, y_train,
    validation_data=(OH_X_test, y_test),
    batch_size=200,
    epochs=50,
    callbacks=[early_stopping], #put callbacks to the list
    verbose=0, #turn of training log
    )
#Now, convert the data to a Pandas dataframe, which makes the plotting easy.
import pandas as pd
#conver the training history to a dataframe
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

