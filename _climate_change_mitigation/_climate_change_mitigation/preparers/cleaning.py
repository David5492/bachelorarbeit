import numpy as np
import pandas as pd
import re
from datetime import datetime
from scipy import stats

# Read data:
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/interim/berlin.csv", na_values=['nan', np.nan])
df = df.drop(['Unnamed: 0'], axis=1)


#Alle Kommas zu Punkten und alle unnützen Punkte weg. Aber nicht bei allen
#Spalten. ich 'save' ein paar. U.a. Grund: Fehler treten auf, wenn als float importiert. 

saver = ['year_built','last_renovated','latitude', 'longitude', 'description', 'equipment', 'description_main', 
            'description_equipment', 'description_location', 'description_misc', 'energy_sources','bedrooms','bathrooms']

df_saver = df[saver]

df = df.drop(saver, axis=1)
df = df.astype('str') 

df = df.apply(lambda x: x.str.replace('.',''))
df = df.apply(lambda x: x.str.replace(',','.'))

df = pd.concat([df, df_saver],axis=1).reset_index()


# Aus 'Floor' mehrere Spalten machen. floor_max ist oft NaN. Dafür -1.

subset = df['floor'].str.split(' ', expand=True)

subset.columns = ['floor_act', 'useless', 'floor_max']
subset['floor_max'].fillna(value=np.nan, inplace=True)
subset = subset.drop('useless', axis=1).astype(float)             

df = pd.concat([df, subset], axis=1).drop('floor', axis=1)


# Zielvariable erstellen und formatieren: 'energy_consumption_value'

subset = df['energy_consumption_value'].str.split(' ', expand=True)
subset.columns = ['energy_consumption', 'useless']
subset.energy_consumption = subset.energy_consumption.astype(float)
subset = subset.drop('useless', axis=1)

df = pd.concat([df, subset], axis=1).drop(['energy_consumption_value'], axis=1)


# Die Spalte 'city-code' erstellen aus'Address_2' und die dann fallen lassen.

subset = df['address_2'].str.split(' ', expand=True)
df['city_code'] = subset[0].astype(int)
df = df.drop('address_2', axis=1)


# Total unnütze Spalten droppen

df = df.drop(['index', 'expose_id', 'city', 'title', 'address_1'], axis=1)


# Alle Text-Cols auf Dopplungen Testen und bereinigen als Prep für NLP:

def adder(s1, s2):
    '''
    Öffnet eine leere Liste.
    Rechnet mit 2 pd.Series. Prüft diese Zeilenweise, ob sie identisch sind.
    Wenn ja, kopiert sie den Inhalt von der ersten Serie in die leere
    Serie.
    Wenn nein, kopiert sie den Inhalt der ersten plus zweiten Serie in
    dei leere Liste.
    Wenn es keine Inhalte gibt, kommt ein Nan-Wert an dem Index in die
    leere Serie.
    Der Rückgabewert ist die vormals leere Liste als Serie.
    '''
    liste = []

    checker = s1 == s2  # boolean filter
    nan_checker_s1 = s1.isna()

    for i in range(len(checker)):
        if checker[i]:  # equals checker[i] == True

            if nan_checker_s1[i]:
                liste.append(s2[i])
            else:
                liste.append(s1[i])

        else:
            liste.append(s1[i] + s2[i])

    serie = pd.Series(liste).astype(str)

    return (serie)

cols_with_duplicates = [('description', 'description_main'), #Liste durch händisches prüfen ermittelt. 
                        ('equipment', 'description_equipment')]

for i, j in cols_with_duplicates:
    df[i + '_clear'] = adder(df[i], df[j])
    df = df.drop([i, j], axis=1)




# Categorials finden und aggregieren. Alle datentypen korrigieren.

# Categorie 1: Parking:

            # todo: regex benutzen. 
            # regex = re.compile(r'(\d*) *(.*)')
            # ss = df['parking']
            # final = pd.DataFrame(index=range(len(ss)), columns=['parking_spaces', 'parking_kind'])

            # dummy = np.arange(0, 1, len(df['deposit'])*2)
            # array = np.array(dummy)
            # array.reshape(len(df['deposit']),2)

#Split: #plätze / art des Platzes. Ist super kompliziert. 



subset1 = df['parking'].str.split(' ', expand=True)
subset1.columns = ['parking_spaces','parking_kind']



add_1 = {'Tiefgaragen-Stellplatz': '1 Tiefgaragen-Stellplatz',
        'Außenstellplatz':'1 Außenstellplatz',
        'Garage':'1 Garage',
        'Parkhaus-Stellplatz':'1 Parkhaus-Stellplatz',
        'Duplex-Stellplatz':'1 Duplex-Stellplatz',
        'Carport':'1 Carport'}

subset1['parking_spaces_copy'] = subset1['parking_spaces'].map(add_1)


subset2 = subset1['parking_spaces_copy'].str.split(' ', expand=True)
subset2.columns = ['parking_spaces_1','parking_kind_1']

test_filter = []

for i in subset1['parking_spaces']:
    j = len (str(i))
    if j < 4:
        x = True
        test_filter.append(x)
    else:
        x = False
        test_filter.append(x)


subset2['parking_spaces_2'] = subset1['parking_spaces'][test_filter]
subset2['parking_spaces_2'].replace('nan', np.nan, inplace=True)

singular = {
        'Tiefgaragen-Stellplatz': 'Tiefgaragen-Stellplatz',
        'Außenstellplatz':'Außenstellplatz',
        'Garage':'Garage',
        'Parkhaus-Stellplatz':'Parkhaus-Stellplatz',
        'Duplex-Stellplatz':'Duplex-Stellplatz',
        'Carport':'Carport',
        'Stellplatz':'Stellplatz',
        'Stellplätze':'Stellplätze',
        'Außenstellplätze':'Außenstellplatz',
        'Garagen':'Garage',
        'Duplex-Stellplätze':'Duplex-Stellplatz'}

subset2['parking_kind_2'] = subset1['parking_kind'].map(singular)

subset2.replace(np.nan, '0', regex = True, inplace = True)
subset2['parking_spaces_1'] = pd.to_numeric(subset2['parking_spaces_1'])
subset2['parking_spaces_2'] = pd.to_numeric(subset2['parking_spaces_2'])

kind_val = {0:0,
            'Tiefgaragen-Stellplatz' : 1,
            'Außenstellplatz' : 2,
            'Garage': 3,
            'Parkhaus-Stellplatz':4,
            'Duplex-Stellplatz': 5,
            'Carport': 6
            }

subset2['parking_kind_1'] = subset2['parking_kind_1'].map(kind_val)
subset2['parking_kind_2'] = subset2['parking_kind_2'].map(kind_val)

#okay, ich hab erfolgreich klassen in zahlen übersetzt. 

subset2.replace(np.nan, 0, regex = True, inplace = True)
subset3 = subset2['parking_spaces_1']+subset2['parking_spaces_2']
subset3.columns = ['parking_spaces']
subset3.replace(0, np.nan, regex = True, inplace = True)

subset4 = subset2['parking_kind_1'] + subset2['parking_kind_2']

subset4.columns = ['parking_kind']
subset4.replace(0, np.nan, regex = True, inplace = True)



val_kind = {0.0:0,
            1.0: 'Tiefgaragen-Stellplatz',
            2.0: 'Außenstellplatz',
            3.0: 'Garage',
            4.0: 'Parkhaus-Stellplatz',
            5.0: 'Duplex-Stellplatz',
            6.0: 'Carport'
            }

subset5 = subset4.map(val_kind)
subset5.columns = ['parking_kind']

final = pd.concat([subset3, subset5], axis = 1)
final.columns = ['parking_spaces', 'parking_kind']

df = pd.concat([df,final], axis = 1)
df = df.drop('parking', axis = 1)


# Categorial 2: 'energy_sources' - Rausgenommen, weil wichtige Var...bzw eine mit 'ner erwähnenswerten Korrelation. 
# energy_sources aufräumen, indem die Doppelten Energiequelllen gedropped werden. 

# filter_list = ['Gas, Fernwärme', 'Gas, Öl', 'Erdwärme, Gas', 'Öl, Fernwärme', 'Gas, Erdgas leicht']

# for value in filter_list:
#     for row in range(len(df)):
#         if df.loc[row, 'energy_sources'] == value:
#             df.drop([row], inplace=True)
#     df=df.reset_index(drop=True)


# Categorial 3: ground_plan -> Binär machen. Verworfen.
# binary = {'yes': 1,
#           'no': 0}

# df['ground_plan'] = df['ground_plan'].map(binary)


# drop von 'energy' und 'energy_certificate'. Grund: Keine Annonce hat kein Zertifikat weil Gesetz
df = df.drop(['energy', 'energy_certificate'], axis=1)


# 'hot_water' zu 'hot_water_included'
binary = {'Energieverbrauch für Warmwasser enthalten': 'yes',
          'nan': 'no'} 

df['hot_water_included'] = df['hot_water'].map(binary)
df = df.drop(['hot_water'], axis=1)


# 'deposit' cleanen: ICH SCHMEIßE ES RAUS. Ist stark mit 'rent' correliert
df = df.drop(['deposit'], axis=1)


# 'nan' zu np.nan. Da ging irgendwo was schief. 
df.replace('nan',np.nan, inplace=True)

# Categorials, Numericals und reiner Text identifizieren (by hand)

text = ['description_misc','description_clear','equipment_clear']

categorials = ['type','pets','condition','quality_of_appliances',
                'heating_type','energy_certificate_type','ground_plan',
                'energy_sources','parking_kind','hot_water_included',
                'city_code']

numericals = ['rent','utilities_cost','heating_cost','cost_total',
                'area','rooms','bedrooms','bathrooms','year_built',
                'last_renovated','latitude','longitude','floor_act',
                'floor_max','energy_consumption','parking_spaces']


# Fillna w/ median & Kill numeric outliers: 
df_numeric = df[numericals].apply(pd.to_numeric, errors='coerce').fillna(df.median()) 

df_num_imputed_clean_all = df[(np.abs(stats.zscore(df_numeric, nan_policy='omit')) < 5).all(axis=1)]
df_num_imputed_clean_nums = df_num_imputed_clean_all[numericals].apply(pd.to_numeric, errors='coerce')

df_num_imputed_clean_nums[numericals] = df_num_imputed_clean_nums[numericals].fillna(df_num_imputed_clean_nums.median())


# Transform nums to integer if possible:
integers = ['bedrooms','bathrooms','year_built','last_renovated','floor_act','floor_max','parking_spaces']

for col in integers:
    df_num_imputed_clean_nums[col] = df_num_imputed_clean_nums[col].astype(int)

df_num_imputed_clean_all[numericals] = df_num_imputed_clean_nums[numericals]


#reset index
df_num_imputed_clean_all = df_num_imputed_clean_all.reset_index().drop(['index'], axis=1)


# Save clean and partially imputed data:
df_num_imputed_clean_all.to_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean.csv", index = False)
print(df.info())
print(df.energy_certificate_type.value_counts())