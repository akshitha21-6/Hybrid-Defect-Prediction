import pandas as pd

def load_data(file):
    df = pd.read_csv(file)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y