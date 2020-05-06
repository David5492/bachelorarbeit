import numpy as np
import pandas as pd

#read Data:

df1 = pd.read_pickle("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/raw/immode_Berlin.pkl")
df2 = pd.read_pickle("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/raw/immode_Berlin2.pkl")

#Concat
df = pd.concat([df1,df2])

#get rid of the doubled data: .drop_duplicates([subset, keep, inplace])
df = df.drop_duplicates() # uses by default all Cols. Could use a subset =.

#reset the index
df = df.reset_index()

#correct colnames: 'ExposeID' zu 'Expose_ID' & normalize name format

df.columns = ['index', 'Expose_ID', 'City', 'Title', 'Description', 'Address_1',
       'Address_2', 'Latitude', 'Longitude', 'Floor', 'Type', 'Rent',
       'Utilities Cost', 'Heating Cost', 'Cost Total', 'Deposit', 'Area',
       'Rooms', 'Bedrooms', 'Bathrooms', 'Energy', 'Equipment', 'Parking',
       'Pets', 'Year built', 'Last Renovated', 'Condition',
       'Quality of Appliances', 'Heating type', 'Energy sources',
       'Energy certificate', 'Energy certificate type',
       'Energy consumption value', 'Hot water', 'Description Main',
       'Description Equipment', 'Description Location', 'Description Misc',
       'Ground Plan']

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#Drop rows w/ NaN in 'energy_consumption_value'
df = df.dropna(subset = ['energy_consumption_value'])
df = df.reset_index()

#Drop absolutely useless columns
df = df.drop(['index','level_0'], axis = 1)

#save:
df.to_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/interim/berlin.csv")
