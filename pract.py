# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 18:20:18 2020

@author: UMAIR
"""

import  numpy as np
import  pandas as pd
import  matplotlib.pyplot as plt

full_dataset = pd.read_csv('train.csv', index_col = 'PassengerId')
full_dataset_test = pd.read_csv('test.csv', index_col = 'PassengerId')

full_dataset['Sex'] = full_dataset['Sex'].map({'male' : 1, 'female' : 0})

# model preparation

y = full_dataset.Survived # with missing data

full_dataset.drop(['Survived'], axis = 1, inplace = True) # Removing label

X = full_dataset.select_dtypes(exclude = ['object']) # Keeping only numerals


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, random_state = 0)

# Imputation
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train[:,3:4]))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

# Model fitting and predictions
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, confusion_matrix

rf_classifier = RandomForestClassifier(n_estimators = 100, random_state = 0)
rf_classifier.fit(imputed_X_train, y_train)
y_pred = rf_classifier.predict(imputed_X_valid)

print(mean_absolute_error(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))

pred_df = pd.DataFrame({'ID': imputed_X_valid.index, 'Survival': y_pred})

# getting the test set
full_dataset_test['Sex'] = full_dataset_test['Sex'].map({'male':1, 'female':0})
full_dataset_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis = 1, inplace = True)
imputed_X_test = pd.DataFrame(my_imputer.transform(full_dataset_test))
imputed_X_test.columns = full_dataset_test.columns
imputed_X_test.index = full_dataset_test.index
test_pred = rf_classifier.predict(imputed_X_test)

test_pred_df = pd.DataFrame({'PessengerID':imputed_X_test.index, 'Survival':test_pred})
