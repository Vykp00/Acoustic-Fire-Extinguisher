# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 16:38:01 2022

@author: Vy

This is my practices with catagorical values
Using acoustic extinguisher fire dataset, 
we will transform catagorical variables
To prepare for machine learning
"""

import pandas as pd
from sklearn.model_selection import train_test_split

#Read the data
data = pd.read_csv("C:/Users/hello/OneDrive/Documents/AI_kaggle/Acoustic_Extinguisher_Fire_Dataset/A_E_Fire_Dataset.csv", delimiter=',')

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
                       and X_train[cname].dtype == "object"]

#Select numerical columns
numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64','float64']]

#These code keep selected columns only but since the data only have number and object type
#These code is unnecessary at the momment
#my_col = low_carniality_cols + numerical_cols
#X_train_1 = X_train[my_col].copy()
#X_test_1 = X_test[my_col].copy()

print(X_train.head())

# Get list of categorical variables
s = (X_train.dtypes == 'object') #This give True/False result of object type columns
object_cols = list(s[s].index) #Colect index data of the column with categorical variables

print("Categorical variables: ", object_cols)

"""
Now we use score_dataset to measure 3 different approaches dealing 
with categorical variable
Report mean absolute error (MAE) from a random forest model. 
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# Approach 1: Droping variable
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_test = X_test.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variable):")
print(score_dataset(drop_X_train, drop_X_test, y_train, y_test))

# Approach 2: Ordinal encoding
from sklearn.preprocessing import OrdinalEncoder

#Make copy to avoid changing original data
label_X_train = X_train.copy()
label_X_test = X_test.copy()

#Apply ordinal encoder to each columns wit categorical data
ordinal_model = OrdinalEncoder()
label_X_train[object_cols] = ordinal_model.fit_transform(X_train[object_cols])
label_X_test[object_cols] = ordinal_model.transform(X_test[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):")
print(score_dataset(label_X_train, label_X_test, y_train, y_test))

# Approach 3: One-Hot encoding
"""
We set handle_unknown='ignore' to avoid errors when the validation data 
contains classes that aren't represented in the training data, and
setting sparse=False ensures that the encoded columns are returned as a numpy array 
(instead of a sparse matrix).
"""
from sklearn.preprocessing import OneHotEncoder

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

print("MAE from Aproach 3 (One hot encoding):")
print(score_dataset(OH_X_train, OH_X_test, y_train, y_test))

"The lowest MAE is approach 3: One hot encoding. So it's the best option."