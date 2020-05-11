import numpy as np
import pandas as pd
import re
from datetime import datetime
from scipy import stats
from text_prep import adder

from datetime import datetime

start = datetime.now()

# Read data:
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/interim/berlin.csv", na_values=['nan', np.nan])
df = df.drop(['Unnamed: 0'], axis=1)


#Alle Kommas zu Punkten und alle unnützen Punkte weg. Aber nicht bei allen
#Spalten. ich 'save' ein paar. U.a. Grund: Fehler treten auf, wenn als float importiert. 

saver = ['year_built','last_renovated','latitude', 'longitude', 'description', 'equipment', 'description_main', 
        'description_equipment', 'description_location', 'description_misc', 'energy_sources','bedrooms','bathrooms']

# df_saver = df[saver]

# df = df.drop(saver, axis=1)
# df = df.astype('str') 

# df = df.apply(lambda x: x.str.replace('.',''))
# df = df.apply(lambda x: x.str.replace(',','.'))

# df = pd.concat([df, df_saver],axis=1).reset_index()


# Aus 'Floor' mehrere Spalten machen. 

subset = df['floor'].str.split(' ', expand=True)
subset.columns = ['floor_act', 'useless', 'floor_max']
subset = subset.drop('useless', axis=1).astype(float)             

df = pd.concat([df, subset], axis=1).drop('floor', axis=1)


# Zielvariable erstellen und formatieren: 'energy_consumption_value'

df['energy_consumption'] = df['energy_consumption_value'].str.strip(" kWh/(m²*a)").str.replace(',','.').astype('float')
df = df.drop(['energy_consumption_value'],axis=1)



# Die Spalte 'city-code' erstellen aus'Address_2' und die dann fallen lassen.

subset = df['address_2'].str.split(' ', expand=True)
df['city_code'] = subset[0].astype(int)
df = df.drop('address_2', axis=1)



# Alle Text-Cols auf Dopplungen Testen und bereinigen als Prep für NLP:

cols_with_duplicates = [('description', 'description_main'), #Liste durch händisches prüfen ermittelt. 
                        ('equipment', 'description_equipment')]

for i, j in cols_with_duplicates:
    df[i + '_clear'] = adder(df[i], df[j]) #funktion in anderem Skript: text_prep.py
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


# Categorial 2: 'energy_sources', indem die doppelten Energiequelllen gedropped werden. 

filter_object = df.energy_sources.str.contains('Gas, Fernwärme|Erdwärme, Gas|Gas, Öl|Öl, Fernwärme|Gas, Erdgas leicht', na=False, regex = True)
df.drop(df[filter_object].index, axis=0, inplace=True)


# Categorial 3: Ausprägungen reduzieren
binary = {'yes': 'yes'}
df['ground_plan'] = df['ground_plan'].map(binary)



# 'hot_water' zu 'hot_water_included'
binary = {'Energieverbrauch für Warmwasser enthalten': 'yes'} 

df['hot_water_included'] = df['hot_water'].map(binary)
df = df.drop(['hot_water'], axis=1)


# 'deposit' säubern: Drop. Ist stark mit 'rent' korreliert.
df = df.drop(['deposit'], axis=1)

# Drop useless cols
df.drop(['expose_id','city','address_1'], axis=1, inplace=True)


# fix dtypes and formats (and check the rooms)

text = ['description_misc','description_clear','equipment_clear','description_location', 'title']

categorials = ['type','pets','condition','quality_of_appliances',
                'heating_type','energy_certificate_type','ground_plan',
                'energy_sources','parking_kind','hot_water_included',
                'city_code','energy', 'energy_certificate']

numericals = ['energy_consumption','rent','utilities_cost','heating_cost','cost_total',
                'area','rooms','bedrooms','bathrooms','year_built',
                'last_renovated','latitude','longitude','floor_act',
                'floor_max','parking_spaces']

# print(len(text) + len(categorials) + len(numericals) == len(df.columns))
# print(df.rent.value_counts(dropna=False).head(30))
# print(df.rent.value_counts(dropna=False).tail(30))
# exit() -> passt. 

float_liste = ['rent', 'rooms', 'utilities_cost','heating_cost','cost_total','area']

for col in float_liste: #Format: '1.000,00'. Problem: '1,000'->*1000 & '1000.00'->/100
    df[col] = df[col].str.replace('.','', regex = True).str.replace(',','.', regex = True).astype('float')



for col in categorials:
    df[col] = df[col].astype('category')


# rearange and reindex:

df = pd.concat([df[numericals],df[categorials],df[text]], axis=1)
df = df.reset_index(drop=True)


#NOTIZEN: Also, die Daten sind sauber, aber voller Ausreisser wegen der komma-punkt-formatierungen. Morgen früh beheben. 


# Save clean and partially imputed data for EDA:
df.to_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_clean_EDA.csv", index = False)


print(df.info())
exit()

# =========================


# Further data prepping for modelling


import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# import numpy as np
# import pandas as pd
# nltk.download()

# categorials, numericals & text are already defined

# create dummie_df 
df_categorials_dummies = pd.get_dummies(df[categorials])


# create num_df 
df_num = df[numericals]


# create NLP_df

df_text = df[text].fillna('XXX') # nur testweise

def text_process(annonce):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in annonce if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('german')]

# All cells of text in one col:
df_text['all_cols'] = df_text.iloc[:,1:].apply(lambda x: ''.join(x), axis=1)

# strings to token integer counts #Start: 19:21 Ziel (ca) 19:38
bow_transformer = CountVectorizer(analyzer=text_process, max_df=0.99, min_df=0.01, max_features = 250).fit(df_text['all_cols']) 

#transform all annoces:
annoces_bow = bow_transformer.transform(df_text['all_cols'])

# tfidf-transformer:
tfidf_transformer = TfidfTransformer().fit(annoces_bow)

# transform the bow:
annonces_tfidf = tfidf_transformer.transform(annoces_bow)

# transform sparse matrix to pd.DataFrame
df_former_sparse = pd.DataFrame.sparse.from_spmatrix(annonces_tfidf)

# concat all sub-dfs
df_all = pd.concat([df_num, df_categorials_dummies, df_former_sparse], axis=1)

# save data for modelling:
df_all.to_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_clean_modelling.csv", index = False)


stop = datetime.now()
print(str(stop - start)) #just4fun
