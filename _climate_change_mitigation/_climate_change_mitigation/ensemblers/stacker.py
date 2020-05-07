import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.insert(1, 'C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/_climate_change_mitigation/modellers')
from ElasticNetReg import enReg
from LinReg import linReg
from NNReg import nnReg
from RandomForestReg import randomForestReg
from SVMReg import svmReg
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

# def stacker():
def stacker(X_train, X_test, y_train, y_test):

    # create 'estimators' list of tuples:
    estimators = [
        ('ElasticNetReg', enReg(X_train, X_test, y_train, y_test)),
        ('LinReg', linReg(X_train, X_test, y_train, y_test)),
        ('NNReg', nnReg(X_train, X_test, y_train, y_test)),
        ('RandomForestReg', randomForestReg(X_train, X_test, y_train, y_test)),
        ('SVMReg', svmReg(X_train, X_test, y_train, y_test))
    ]

    # create and train model:
    reg = StackingRegressor(estimators=estimators, 
                            final_estimator = RandomForestRegressor(
                            n_estimators=10,random_state=42))
    reg.fit(X_train, y_train) #.score(X_test, y_test)

    # predicting values:
    y_pred = reg.predict(X_test)

    # My model evaluation: 
    return total_validation(y_test, y_pred)

metric = stacker(X_train, X_test, y_train, y_test)
print(metric)

stop = datetime.now()
print(str(stop - start)) #just4fun

