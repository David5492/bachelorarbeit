import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
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

# def NNReg():
def nnReg(X_train, X_test, y_train, y_test):

    # creating and training
    NNReg = KNeighborsRegressor(n_neighbors=2)
    NNReg.fit(X_train,y_train)

    # predicting values:
    y_pred = NNReg.predict(X_test)

    # My model evaluation: 
    return total_validation(y_test, y_pred)

metric = nnReg(X_train, X_test, y_train, y_test)
print(metric)

stop = datetime.now()
print(str(stop - start)) #just4fun

# Ergebnis: 0:00:03
# Schrott! r-squared ist negativ. 
# Das geht zwar, aber zeigt das H0 besser wäre (straight line)

#                   value
# MAE              25.99
# MAPE (%)         36.33
# MSE            1972.59
# RMSE (Units)     44.41
# r-squared (%)    -8.00