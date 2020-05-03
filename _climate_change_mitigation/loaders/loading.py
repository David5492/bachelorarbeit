import numpy as np
import pandas as pd

#read Data:

df1 = pd.read_pickle("C:/Users/test/_climate_change_mitigation/data/raw/immode_Berlin.pkl")
df2 = pd.read_pickle("C:/Users/test/_climate_change_mitigation/data/raw/immode_Berlin2.pkl")

#Check if data has the same Cols:

print(list(df1)==list(df2)) #True -> concat. Append wäre auch drin gewesen.
df = pd.concat([df1,df2])








#get rid of the doubled data: .drop_duplicates([subset, keep, inplace])

df = df.drop_duplicates() # uses by default all Cols. Could use a subset =.
df = df.reset_index()



#Spaltennamen Korrigieren:

#'ExposeID' zu 'Expose_ID'
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



#Drop der Rows ohne Label für Zielvariable
df = df.dropna(subset = ['energy_consumption_value'])
df = df.reset_index()


#Absoult unnütze Spalte droppen
df = df.drop(['index','level_0'], axis = 1)

print(df.head())
print(df.shape)


print(df.year_built.value_counts(dropna=False))



#Abspeichern:

df.to_csv("C:/Users/test/_climate_change_mitigation/data/interim/berlin.csv", na_rep = -1)


