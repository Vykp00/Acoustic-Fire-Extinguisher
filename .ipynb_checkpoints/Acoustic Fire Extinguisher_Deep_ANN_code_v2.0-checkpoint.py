# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 17:57:42 2022

@author: Vy
"""
"""
This is my sample Deep NN v2.0
Data: Acoustic Fire Extinguisher
Task: Predict the extinguishing status of the fire based on certain levels of sound
    and various factors
Change log:
    - Classes: Binary Classification
    - In the final layer include a 'sigmoid' activation so that the model will produce class probabilities.
"""
#Setup plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
#Set Matplotlib defaults
plt.rc('figure', autolayout= True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

data = pd.read_csv("A_E_Fire_Dataset.csv")

#Select features and target variables
X = data.drop('STATUS', axis=1)
y= data.STATUS

feature_num = ['SIZE','DISTANCE','DESIBEL','AIRFLOW','FREQUENCY']
feature_cat = ['FUEL']

#Set preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), feature_num),
    (OneHotEncoder(), feature_cat),
    )

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

#Preprocess data
#Since the STATUS value is 0 or 1, we don't need to rescale it
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

input_shape = [X_train.shape[1]]

#Define model
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks

model= keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=input_shape),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
    ])

#Set optimizer, loss, and metric
model.compile(
    optimizer= 'adam',
    loss= 'binary_crossentropy',
    metrics=['binary_accuracy'],
    )
early_stopping = callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
    )

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=512,
    epochs=200,
    callbacks=[early_stopping],
)

history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot(title="Cross-entropy")
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot(title="Accuracy")
