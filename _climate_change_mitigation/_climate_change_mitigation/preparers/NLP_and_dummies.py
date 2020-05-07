import numpy as np
import pandas as pd
#nltk.download()
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


#load data
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean.csv")


text = ['description_misc','description_clear','equipment_clear','description_location']

categorials = ['type','pets','condition','quality_of_appliances',
                'heating_type','energy_certificate_type','ground_plan',
                'energy_sources','parking_kind','hot_water_included',
                'city_code']

numericals = ['rent','utilities_cost','heating_cost','cost_total',
                'area','rooms','bedrooms','bathrooms','year_built',
                'last_renovated','latitude','longitude','floor_act',
                'floor_max','energy_consumption','parking_spaces']

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

# strings to token integer counts
bow_transformer = CountVectorizer(analyzer=text_process).fit(df_text['all_cols']) 

#transform all annoces:
annoces_bow = bow_transformer.transform(df_text['all_cols'])

# tfidf-transformer:
tfidf_transformer = TfidfTransformer().fit(annoces_bow)

# transform the bow:
annonces_tfidf = tfidf_transformer.transform(annoces_bow)

# check if all worked:
print(messages_tfidf.shape)


  
