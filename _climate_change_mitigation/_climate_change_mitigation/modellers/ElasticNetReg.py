import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
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

# def ElasticNetReg():
def enReg(X_train, X_test, y_train, y_test):
    # creating and training
    en = ElasticNet(random_state=42)
    en.fit(X_train,y_train)

    # predicting values:
    y_pred = en.predict(X_test)

    # My model evaluation: 
    return total_validation(y_test, y_pred)

metric = enReg(X_train, X_test, y_train, y_test)
print(metric)

stop = datetime.now()
print(str(stop - start)) #just4fun

# Ergebnis: 0:00:01
# Schrott! r-squared ist negativ. 
# Das geht zwar, aber zeigt das H0 besser wäre (straight line)

#                   value
# MAE              29.04
# MAPE (%)         39.98
# MSE            1855.53
# RMSE (Units)     43.08
# r-squared (%)    -1.60