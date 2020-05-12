# intendet steps:
    # 1. Impute Missing Values
    # 2. Transform vía scipy.stats.boxcox(data, lmbda=) (spare ich mir)
    # 3. Outliers vía scipy.stats.mstats.winsorize(limits=[0.05,0.05])
    # 4. Scaling vía StandardScaler() und MinMaxScaler()
    # 5. NLP

import numpy as np
import pandas as pd
import re
from scipy import stats
import string
from nltk.corpus import stopwords
# nltk.download()
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import random
from sklearn.impute import SimpleImputer
from scipy.stats.mstats import winsorize
from nltk.stem.porter import PorterStemmer

random.seed(42)

from datetime import datetime

start = datetime.now()

# Load data
df= pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_clean_EDA.csv")

# define subset lists
text = ['description_misc','description_clear','equipment_clear','description_location', 'title']

categorials = ['type','pets','condition','quality_of_appliances',
                'heating_type','energy_certificate_type','ground_plan',
                'energy_sources','parking_kind','hot_water_included',
                'city_code','energy', 'energy_certificate']

numericals = ['energy_consumption','rent','utilities_cost','heating_cost','cost_total',
                'area','rooms','bedrooms','bathrooms','year_built',
                'last_renovated','latitude','longitude','floor_act',
                'floor_max','parking_spaces']


# create sub dfs:
df_categorials_dummies = pd.get_dummies(df[categorials])
df_num = df[numericals]
df_text = df[text] #.fillna('XXX') # nur testweise


# Impute missing num 

median_impute = ['utilities_cost','heating_cost','latitude','longitude','floor_act',
                'floor_max','parking_spaces']
most_frequent_impute = ['bedrooms','bathrooms','year_built','last_renovated',]

for col in median_impute:
    imp_median = SimpleImputer(strategy='median')
    df_num[col] = imp_median.fit_transform(df_num[[col]])

for col in most_frequent_impute:
    imp_mf = SimpleImputer(strategy='most_frequent')
    df_num[col] = imp_mf.fit_transform(df_num[[col]])


# Impute missing text 

def text_imputer(text):
    """
    Rechnet mit einem str.
    Prüft ob ein Wert nan ist und ersetzt ihn gegebenfalls mit Kauderwelsch.
    Sinn: NLP funktioniert sonst nicht und bei einem anderen Impute entsteht
    ein Bias. 
    """
    if pd.isnull(text):
        return ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=random.randint(50,100)))
    else:
        return text

for col in df_text.columns: 
    df_text[col] = df_text[col].apply(text_imputer)


# Kill outliers by setting upper and lower limits

for col in df_num.columns: 
    masked_array = winsorize(df_num[col], limits =[0.005,0.005])
    df_num[col] = pd.DataFrame(masked_array, columns = [col])


# NLP:

def text_process(annonce):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Produces word stems
    4. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in annonce if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # remove any german stopwords
    nostop = [word for word in nopunc.split() if word.lower() not in stopwords.words('german')]

    # Reduce words to their stem
    porter = PorterStemmer()
    return [porter.stem(word) for word in nostop]

# All cells of text in one col:
df_text['all_cols'] = df_text.iloc[:,1:].apply(lambda x: ''.join(x), axis=1)

# strings to token integer counts
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
df = df.reset_index(drop=True)

# save data for modelling:
df_all.to_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_preprocessed.csv", index = False)


stop = datetime.now()
print(str(stop - start)) #just4fun

# Time: 00:23:20