








####      WIRD NICHT MEHR VERWENDET. DATEN WERDEN DADURCH TEILWEISE ZU EINHEITSBREI      ####











# I already did the numerical imputaion in the cleaning-part, since I had to, bc I wanted to kill the outliers. 

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


df = pd.read_csv("C:/Users/test/_climate_change_mitigation/data/processed/berlin_num_imputed_clean.csv")

# NaN der Categorials via KNNImputer auffüllen

# Prep data
categorials = ['type','pets','condition','quality_of_appliances',
                'heating_type','energy_certificate_type','ground_plan',
                'energy_sources','parking_kind','hot_water_included',
                'city_code']


df_categorials = df[categorials]

# train modell
impute_category = SimpleImputer( strategy='most_frequent')
impute_category.fit(df_categorials)

# fill NaNs in array, make it to a df. 
imputed_train_array = impute_category.transform(df_categorials)
imputed_train_df = pd.DataFrame(imputed_train_array, columns = categorials)

# concat
df = df.drop(categorials, axis=1)
df = pd.concat([df, imputed_train_df], axis=1)

# save
df.to_csv("C:/Users/test/_climate_change_mitigation/data/processed/berlin_all_imputed_clean.csv", index = False)
