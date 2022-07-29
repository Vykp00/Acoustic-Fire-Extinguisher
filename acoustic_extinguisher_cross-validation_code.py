# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 13:43:26 2022

@author: Vy
"""
"""
This is my practices of using cross-validation
Data: Acoustic-Fire-Extinguisher 
Goal: Be able to handle realistic situation
Tune the ML model with cross validation for both numerical and categorical data

"""
#Set up code checking
import os
if not os.path.exists("A_E_Fire_Dataset.csv"):
    os.symlink("C:/Users/hello/OneDrive/Documents/AI_kaggle/Acoustic_Extinguisher_Fire_Dataset/A_E_Fire_Dataset.csv", "A_E_Fire_Dataset.csv")
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex5 import *
print("Setup Complete")    
    
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv("A_E_Fire_Dataset.csv")
    

# Seperate target from predictors
y = data.STATUS
X = data.drop(['STATUS'], axis=1)

# Check for missing data
missing_val_count_by_columns = (X.isnull().sum())
print("Missing value by columns:")
print(missing_val_count_by_columns[missing_val_count_by_columns > 0])

"""
Transformer and Pipeline require index types instead of list.
So use X.columns or vars = [1,2,3] when selecting numerical or categorical features
Or just use num_cols and cat_cols below
"""
# Select numerical data
num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
# X_num_vars = X[num_cols].copy().columns

# Select catergorical data
cat_cols = [cname for cname in X.columns if X[cname].dtype in ['object']]
# X_cat_vars = X[cat_cols].copy().columns

# To make the code more straightforward, use pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Preprocessing for numerical and categorical data
num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse=(False))

#Building preprocessor for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_cols),
                                               ("cat", cat_transformer, cat_cols)], verbose_feature_names_out=(True))

#Define RandomForest Model with Pipeline
my_pipeline= Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', RandomForestRegressor(n_estimators=100, random_state=10))])
"""
We obtain the cross-validation scores with the cross_val_score() 
function from scikit-learn. We set the number of folds with the cv parameter.
"""
from sklearn.model_selection import cross_val_score
# Multiply by -1 since sklean calculare *negative* MAE
score = (-1) * cross_val_score(my_pipeline, X, y,
                             cv=3, scoring= 'neg_mean_absolute_error')
print("MAE scores:\n", score)

"""
Scikit-learn has a convention where all metrics are defined so a high number is better. 
Using negatives here allows them to be consistent with that convention, though negative MAE 
is almost unheard of elsewhere.

We typically want a single measure of model quality to compare alternative models. 
So we take the average across experiments.
"""
print("Average MAE score (across experiments):", score.mean())

"""
In this exercise, you'll use cross-validation to select parameters for a machine learning model.

Begin by writing a function `get_score()` that reports the average (over three cross-validation folds) MAE of a machine learning pipeline that uses:
- the data in `X` and `y` to create folds,
- `SimpleImputer()` (with all parameters left as default) to replace missing values, and
- `RandomForestRegressor()` (with `random_state=0`) to fit a random forest model.

The `n_estimators` parameter supplied to `get_score()` is used when setting the number of trees in the random forest model. 
"""
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    #Multiply by -1 since sklearn calculate *negative* MAE
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', RandomForestRegressor(n_estimators, random_state=0))])
    scores = -1 * cross_val_score(my_pipeline, X, y, cv=3, scoring= 'neg_mean_absolute_error')
    return scores.mean()
"""
Evaluate the model performance corresponding to eight different values for 
the number of trees in the random forest: 50, 100, 150, ..., 300, 350, 400.

Store your results in a Python dictionary results, where results[i] is the 
average MAE returned by get_score(i).
"""
results = {}
for i in range(1,9):
    results[50*i] = get_score(i)
print(results)

#Now we visualize the result
import matplotlib.pyplot as plt

plt.plot(list(results.keys()),list(results.values()))
plt.show()