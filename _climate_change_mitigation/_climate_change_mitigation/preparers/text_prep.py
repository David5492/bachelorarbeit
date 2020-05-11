# from difflib import SequenceMatcher

# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()

# print(similar("Ich mag Züge","Ich mag dich"))

#Das is es. Damit finde ich heraus, ub die Texteinträge nur knapp verschieden sind.

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


def adder(s1, s2):
    '''
    Öffnet eine leere Liste.
    Rechnet mit 2 pd.Series. Prüft diese Zeilenweise, ob sie sehr ähnlich sind.
    Wenn ja, kopiert sie den Inhalt von der ersten Serie in die leere
    Liste.
    Wenn nein, kopiert sie den Inhalt der ersten plus zweiten Serie in
    dei leere Liste.
    Wenn es keine Inhalte gibt, kommt ein Nan-Wert an dem Index in die
    leere Serie.
    Der Rückgabewert ist die vormals leere Liste als Serie.
    '''
    s1 = s1.fillna('XXX')
    s2 = s2.fillna('XXX')
    
    def similar(s1, s2):
        '''
        Rechnet mit 2 Series. 
        Vergleicht die Ähnlichkeit der str-Abfolge von 0 bis 1 (1 = 100%).
        Gibt einen Array der Ähnlichkeiten aus. 
        '''
        check_liste = []
        for row in range(len(s1)):
            check_liste.append(SequenceMatcher(None, s1[row], s2[row]).ratio())
        
        return np.array(check_liste)

    output_liste = []

    # checker = s1 == s2  # boolean filter
    checker = similar(s1,s2) > 0.8
    nan_checker_s1 = s1 == 'XXX'
    nan_checker_s2 = s2 == 'XXX'

    for i in range(len(checker)):
        
        if checker[i]: 

            if (nan_checker_s1[i]) & (nan_checker_s2[i]):
                output_liste.append('XXX')
            elif nan_checker_s1[i]:
                output_liste.append(s2[i])
            else:
                output_liste.append(s1[i])

        else:
            output_liste.append(s1[i] + ' ' + s2[i])

    return pd.Series(output_liste).astype(str)




