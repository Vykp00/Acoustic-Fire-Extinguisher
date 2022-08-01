# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 13:42:02 2022

@author: Vy
"""
"""
This is my practice with pipelines
Similar to how a pipeline bundles together preprocessing 
and modeling steps, we use the "ColumnTransformer" class to 
bundle together different preprocessing steps. The code below:
    
+ imputes missing values in numerical data, and
+ imputes missing values and applies a one-hot encoding to categorical data.
Data: Acoustic Fire Extinguisher 

"""
import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data
data = pd.read_csv("A_E_Fire_Dataset.csv", delimiter=',')

#Separate target from predictors
y = data.STATUS
X = data.drop(['STATUS'], axis=1)

#Divide data into training and validation subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

#Check for missing values
print(X.shape)
missing_val_count_by_column = (X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_carniality_cols = [cname for cname in X_train.columns if X_train[cname].nunique() < 10 
                       a
                       

#Select numerical columns
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']]

#These code keep selected columns only but since the data only have number and object type
#These code is unnecessary at the momment
#my_col = low_carniality_cols + numerical_cols
#X_train_1 = X_train[my_col].copy()
#X_test_1 = X_test[my_col].copy()



# Step 1: Define preprocessing step
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
num_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])
# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[('num', num_transformer, numerical_cols),
                 ('cat', cat_transformer, low_carniality_cols)])

# Step 2: Define RandomForestRegressor Model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=10)

# Step 3: Create and Evaluate the model
"""
With the pipeline, we preprocess the training data and fit the model in a single line of code.
 (In contrast, without a pipeline, we have to do imputation, one-hot encoding, and model training in separate steps.
  This becomes especially messy if we have to deal with both numerical and categorical variables!)
With the pipeline, we supply the unprocessed features in X_valid to the predict() command, and the pipeline 
    automatically preprocesses the features before generating predictions. (However, without a pipeline, we have to 
    remember to preprocess the validation data before making predictions.)
"""

from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])

# Preprocessing of training data, fit model
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
y_pred = my_pipeline.predict(X_test)
print(X_train.head())
# Evaluate the model
score = mean_absolute_error(y_test, y_pred)
print("MAE: ", score)

