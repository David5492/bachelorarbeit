import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from validation_metric import total_validation

#load data
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean.csv")

# drop text-cols
text = ['description_misc','description_clear','equipment_clear']
df = df.drop(text, axis=1)

# reanrange labels in 1. col
df = df[['energy_consumption', 'type', 'rent', 'utilities_cost', 'heating_cost', 'cost_total', 'area',
    'rooms', 'pets', 'condition', 'quality_of_appliances', 'heating_type',
    'energy_certificate_type', 'ground_plan', 'year_built',
    'last_renovated', 'latitude', 'longitude',
    'energy_sources', 'bedrooms', 'bathrooms',
    'floor_act', 'floor_max', 'city_code',
    'parking_spaces','parking_kind', 'hot_water_included']]

#build dummies
categorials = ['type','pets','condition','quality_of_appliances',
                'heating_type','energy_certificate_type','ground_plan',
                'energy_sources','parking_kind','hot_water_included',
                'city_code']

df_cat_dummies = pd.get_dummies(df[categorials])
df_just_nums = df.drop(categorials, axis = 1)

df = pd.concat([df_just_nums, df_cat_dummies], axis = 1)


# split data
X = df.iloc[:,1:]
y = df.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# creating and training
lm = LinearRegression()
lm.fit(X_train,y_train)

# predicting values:
y_pred = lm.predict(X_test)

# My model evaluation: 
df_LinReg_validation = total_validation(y_test, y_pred)
print('\n', df_LinReg_validation, '\n')