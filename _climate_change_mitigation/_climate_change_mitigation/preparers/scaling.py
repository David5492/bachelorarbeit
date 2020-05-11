from sklearn.preprocessing import StandardScaler

def scale(df):
    scaler = StandardScaler().fit(df)
    return scaler.transform(df)