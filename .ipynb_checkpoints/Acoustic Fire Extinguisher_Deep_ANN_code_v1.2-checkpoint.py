# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 14:50:20 2022

@author: Vy

This is my sample of Deep NN v1.2
Data: Acoustic Fire Extinguisher
Task: Predict the extinguishing status of the fire based on certain levels of sound
    and various factors
Change log:
    - Preprocess numerical and catergorical data with StandardScaler and OneHotEncoder
    -
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
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

data = pd.read_csv("A_E_Fire_Dataset.csv")

#Select features and target variables
X = data.drop('STATUS', axis=1)
y= data.STATUS

feature_num = ['SIZE','DISTANCE','DESIBEL','AIRFLOW','FREQUENCY']
feature_cat = ['FUEL']

preprocessor = make_column_transformer(
    (StandardScaler(), feature_num),
    (OneHotEncoder(), feature_cat),
    )
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)
#Since the STATUS value is 0 or 1, we don't need to rescale it
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

input_shape = [X_train.shape[1]]
print("Input_shape: {}".format(input_shape))

#Set callbacks
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=5,
    restore_best_weights=True
    )

#Build the model
model = keras.Sequential([
    layers.Dense(1024, activation='relu', input_shape=input_shape),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1)
    ])

model.compile(
    optimizer='adam',
    loss='mae',
    )
#Fit the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=500,
    epochs=100,
    callbacks=[early_stopping],
    verbose=0
    )

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum Validation Loss: {:0.4f}".format(history_df['val_loss'].min()));
