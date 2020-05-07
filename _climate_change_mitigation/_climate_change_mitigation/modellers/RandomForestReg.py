import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from validation_metric import total_validation

from datetime import datetime

start = datetime.now()

#load data
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean_modelling.csv")

# fill NaN:
df = df.fillna(df.mean())

# split data
X = df.iloc[:,1:]
y = df.iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# def RandomForestReg():
def randomForestReg(X_train, X_test, y_train, y_test):

    # train model:
    regr = RandomForestRegressor(max_depth=20, random_state=42)
    regr.fit(X_train, y_train)

    # predict values:
    y_pred = regr.predict(X_test)

    # validate
    return total_validation(y_test, y_pred)

metric = randomForestReg(X_train, X_test, y_train, y_test)
print(metric)

# Ich will die optimale tiefe eines Baumes erfahren und probiere daher herum
# -> Dafür eigenes Notebook

stop = datetime.now()
print(str(stop - start)) #just4fun

# Ergebnis: 0:01:26

#                  value
# MAE             18.46
# MAPE (%)        30.53
# MSE            889.05
# RMSE (Units)    29.82
# r-squared (%)   51.32