import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split

def load_split():
    #load data
    df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean_modelling.csv")

    # fill NaN:
    df = df.fillna(df.mean())

    # split data
    X = df.iloc[:,1:]
    y = df.iloc[:,0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #return
    return X_train, X_test, y_train, y_test