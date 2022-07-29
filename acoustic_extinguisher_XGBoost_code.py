# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 15:50:34 2022

@author: Vy
"""
"""
This is my practice to build and optimize models with gradient boosting.
Gradient boosting is a method that goes through cycles to 
iteratively add models into an ensemble.

Goal: +Test 3 different models
    
    + Use XG Boost -extreme gradient boosting, which is an implementation 
of gradient boosting with several additional features focused on performance 
and speed. (Scikit-learn has another version of gradient boosting, 
            but XGBoost has some technical advantages.)

XGBoost is the most accurate modeling technique for structured data (the type 
 of data you store in Pandas DataFrames, as opposed to more exotic types of 
 data like images and videos). With careful parameter tuning, you can train 
 highly accurate models.

Data: Acoustic Fire Extinguisher
"""
import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data
data = pd.read_csv("C:/Users/hello/OneDrive/Documents/AI_kaggle/Acoustic_Extinguisher_Fire_Dataset/A_E_Fire_Dataset.csv")

#Select explainatory variables
X = data.drop(['STATUS'], axis=1)
y = data.STATUS

# First we convert categorial variable using OneHotEncoder
# Get a list of categorical variable
s = (X.dtypes == 'object') #This give True/False result of object type columns
cat_cols = list(s[s].index) #Colect index data of the column with categorical variables


#Apply OneHotEncoder model
from sklearn.preprocessing import OneHotEncoder
OH_model= OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_model.fit_transform(X[cat_cols]))
OH_cols.index = X.index # One-hot encoding removed index; now we put it back
# Get feature name for OH encoder. Without it, feature name is "x0", "x1",.."nfeature" by defaults
num_X = X.drop(cat_cols, axis=1) #Drop old categorical columns
OH_cols.columns = OH_model.get_feature_names_out()
OH_X = pd.concat([num_X, OH_cols], axis=1)

# Seperate data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(OH_X, y, test_size=0.2, random_state=0)

"""
XGBoost has a few parameters that can dramatically affect accuracy and 
training speed. The first parameters you should understand are:

n_estimators
n_estimators specifies how many times to go through the modeling cycle 
described above. It is equal to the number of models that we include in the ensemble.

    +Too low a value causes underfitting, which leads to inaccurate predictions 
        on both training data and test data.
    +Too high a value causes overfitting, which causes accurate predictions on 
        training data, but inaccurate predictions on test data (which is what we care about).
Typical values range from 100-1000, though this depends a lot on the 
learning_rate parameter discussed below.

Early stopping causes the model to stop iterating when the validation score
 stops improving, even if we aren't at the hard stop for n_estimators. 
 It's smart to set a high value for n_estimators and then use early_stopping_rounds
 to find the optimal time to stop iterating.
"""
#Build MODEL 1: random_state =0, other is by defaults
from xgboost import XGBRegressor
#Instead of getting predictions by simply adding up the predictions from each 
# component model, we can multiply the predictions from each model by a small number 
#(known as the learning rate) before adding them in.
my_model_1 = XGBRegressor(random_state=0)
#When using early_stopping_rounds, you also need to set aside some data for 
# calculating the validation scores - this is done by setting the eval_set parameter.
my_model_1.fit(X_train, y_train)
"""
On larger datasets where runtime is a consideration, you can use parallelism to 
build your models faster. It's common to set the parameter n_jobs equal to the 
number of cores on your machine. On smaller datasets, this won't help.

The resulting model won't be any better, so micro-optimizing for fitting time 
is typically nothing but a distraction. But, it's useful in large datasets 
where you would otherwise spend a long time waiting during the fit command.

"""

#Make prediction and evaluate the model
from sklearn.metrics import mean_absolute_error
y_pred_1 = my_model_1.predict(X_test)
print("Mean Absolute Error MODEL 1: " + str(mean_absolute_error(y_pred_1, y_test))) #Result: 0.08327621617352889

"""
These following code test whether we can improve the model
"""
# MODEL 2: Experiment until MAE 2 < MAE1. Create a better model
my_model_2 = XGBRegressor(n_estimators=1000, learning_rate=0.1)
my_model_2.fit(X_train, y_train)
y_pred_2 = my_model_2.predict(X_test)
print("Mean Absolute Error MODEL 2: " + str(mean_absolute_error(y_test, y_pred_2))) #Result: 0.0759303122336413

# MODEL 3: Create and experiment to get the worst model
my_model_3 = XGBRegressor(n_estimators= 50, learning_rate=0.0001)
my_model_3.fit(X_train, y_train)
y_pred_3 = my_model_3.predict(X_test)
print("Mean Absolute Error MODEL 3: " + str(mean_absolute_error(y_test, y_pred_3))) #Result: 0.49808108461521594