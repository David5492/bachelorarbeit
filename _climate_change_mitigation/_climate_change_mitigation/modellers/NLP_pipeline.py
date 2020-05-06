import numpy as np 
import pandas as pd 
import nltk
#nltk.download()
import string
import random
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# set seed:
random.seed(42)

# Load data
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean.csv")
text = ['energy_consumption','equipment_clear','description_clear','description_misc']

df = df[text].fillna('XXX') # nur testweise


# Text Pre-Processing

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

# def MAPE as additional evaluation metric

def mean_absolute_percentage_error(y_test, y_pred): 
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# def tester, which shows me which of the cols have which predicitve value

def tester(df):
    """
    teilt df in df_features und df_labels. Letztere nur mit den labels. 
    macht train-test-split mit jeder col in den features.
    Jagt die daten Durch ein Pipeline, die in einer ridge regression mündet.
    Gibt je col 3 Fehlermaße aus. 
    """
    df_labels = df.iloc[:,0]
    df_features = df.iloc[:,1:]

    for col in df_features.columns:
        # train-test-split:
        X_train, X_test, y_train, y_test = train_test_split(df_features[col], df_labels, test_size=0.3)

        # Creating Pipeline
        pipeline = Pipeline([
            ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
            ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
            ('regressor', Ridge(alpha=0.1)),  # train on TF-IDF vectors w/ ridge regressor
        ])

        # train model
        pipeline.fit(X_train, y_train)

        # predict vía X_test
        y_pred = pipeline.predict(X_test)

        # Evaluate vía y_test
        print(col)
        print('mean_absolute_error: ' + str(mean_absolute_error(y_test, y_pred)))
        print('mean_squared_error: ' + str(mean_squared_error(y_test, y_pred)))
        print('mean_absolute_percentage_error: ' + str(mean_absolute_percentage_error(y_test, y_pred)))
        print('\n')

tester(df)


# Ergebnisse: 

# equipment_clear
# mean_absolute_error: 24.643815984040927
# mean_squared_error: 1217.7704411987957
# mean_absolute_percentage_error: 21.50994447837582


# description_clear
# mean_absolute_error: 19.489137778793165
# mean_squared_error: 861.9524704814194
# mean_absolute_percentage_error: 16.928810336925174


# description_misc
# mean_absolute_error: 26.30191049768944
# mean_squared_error: 1328.4519254751888
# mean_absolute_percentage_error: 27.3149756435473


# Aufgabe für's nächste mal: 

# MAPE einführen (check)
# Noch eine Zusätzliche Spalte an df, welche alle anderen Spalten zeilenweise verbindet. 