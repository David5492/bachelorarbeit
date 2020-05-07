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
from validation_metric import total_validation

# set seed:
random.seed(42)

# Load data
df = pd.read_csv("C:/Users/test/Documents/GitHub/bachelorarbeit/_climate_change_mitigation/data/processed/berlin_num_imputed_clean.csv")

# transform data
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


# def tester, which shows me which of the cols have which predicitve value

def tester(df):
    """
    LABEL MÜSSEN IN DER ERSTEN COL SEIN!!

    Erschafft eine extra-col, die die Inhalte aller anderen Cols enthält.
    Teilt df in df_features und df_labels. Letztere nur mit den labels. 
    macht train-test-split mit den features.
    Jagt die Daten Durch ein Pipeline, die in einer ridge regression mündet.
    Gibt je col 4 Fehlermaße aus: MAE, MAPE, MSE, RMSE. 
    """
    df['all_cols'] = df.iloc[:,1:].apply(lambda x: ''.join(x), axis=1)
    df_labels = df.iloc[:,0]
    df_features = df.iloc[:,1:]

    for col in df_features.columns:
        # train-test-split:
        X_train, X_test, y_train, y_test = train_test_split(df_features[col], df_labels, random_state=42, test_size=0.3)

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
        df_validation_output = total_validation(y_test, y_pred)
        print(df_validation_output, '\n')


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

# all_cols
# mean_absolute_error: 19.358487453777812
# mean_squared_error: 836.8946299593965
# mean_absolute_percentage_error: 30.690176166118583


# Aufgabe für's nächste mal: 

# (check) MAPE einführen 
# (check) Noch eine Zusätzliche Spalte an df, welche alle anderen Spalten zeilenweise verbindet. 

# Conclusion: description_clear ist bester Schätzer. 


# UND NOCHMAL:

# equipment_clear
# mean_absolute_error (MAE): 24.51
# mean_absolute_percentage_error (MAPE): 21.02 %
# mean_squared_error (MSE): 1212.74
# root_mean_squared_error (RMSE): 34.82 kwh/(m^2 * a)


# description_clear
# mean_absolute_error (MAE): 20.13
# mean_absolute_percentage_error (MAPE): 34.31 %
# mean_squared_error (MSE): 931.49
# root_mean_squared_error (RMSE): 30.52 kwh/(m^2 * a)


# description_misc
# mean_absolute_error (MAE): 26.34
# mean_absolute_percentage_error (MAPE): 25.06 %
# mean_squared_error (MSE): 1334.98
# root_mean_squared_error (RMSE): 36.54 kwh/(m^2 * a)


# all_cols
# mean_absolute_error (MAE): 19.95
# mean_absolute_percentage_error (MAPE): 17.0 %
# mean_squared_error (MSE): 907.4
# root_mean_squared_error (RMSE): 30.12 kwh/(m^2 * a)


# Aufgabe: 
# (check) random_state=42 verwenden!

# Und ein letztes Mal: 

# equipment_clear
# MAE: 24.5
# MAPE: 24.09 %
# MSE: 1210.44
# RMSE: 34.79 kwh/(m^2 * a)


# description_clear
# MAE: 19.95
# MAPE: 19.58 %
# MSE: 930.83
# RMSE: 30.51 kwh/(m^2 * a)


# description_misc
# MAE: 25.77
# MAPE: 26.51 %
# MSE: 1283.73
# RMSE: 35.83 kwh/(m^2 * a)


# all_cols
# MAE: 19.08
# MAPE: 18.84 %
# MSE: 833.31
# RMSE: 28.87 kwh/(m^2 * a)



# Nochmal:

# equipment_clear
# MAE: 24.5
# MAPE: 24.09 %
# MSE: 1210.44
# RMSE: 34.79 kwh/(m^2 * a)
# r-squared: 0.22


# None


# description_clear
# MAE: 19.95
# MAPE: 19.58 %
# MSE: 930.83
# RMSE: 30.51 kwh/(m^2 * a)
# r-squared: 0.4


# None


# description_misc
# MAE: 25.77
# MAPE: 26.51 %
# MSE: 1283.73


# None


# all_cols
# MAE: 19.08
# MAPE: 18.84 %
# MSE: 833.31
# RMSE: 28.87 kwh/(m^2 * a)
# r-squared: 0.46


# None

#Neue Aufgabe: Ausgabe als Matrix. 

# equipment_clear
#                     0
# MAE             24.50
# MAPE (%)        24.09
# MSE           1210.44
# RMSE (Units)    34.79
# r-squared        0.22
# None
# description_clear
#                    0
# MAE            19.95
# MAPE (%)       19.58
# MSE           930.83
# RMSE (Units)   30.51
# r-squared       0.40
# None
# description_misc
#                     0
# MAE             25.77
# MAPE (%)        26.51
# MSE           1283.73
# RMSE (Units)    35.83
# r-squared        0.17
# None
# all_cols
#                    0
# MAE            19.08
# MAPE (%)       18.84
# MSE           833.31
# RMSE (Units)   28.87
# r-squared       0.46
# None


#welches alpha=? Hier gleich 1:

# equipment_clear
#                 value
# MAE             24.86
# MAPE (%)        25.54
# MSE           1136.97
# RMSE (Units)    33.72
# r-squared        0.27

#welches alpha=? Hier gleich 0 (also einfache linreg): SCHROTT

# equipment_clear 
#                      value
# MAE           4.998057e+20
# MAPE (%)      4.874213e+20
# MSE           7.361424e+41
# RMSE (Units)  8.579874e+20
# r-squared    -4.755639e+38

# FAZIT: ICH BLEIBE ERSTMAL BEI alpha = 0.1