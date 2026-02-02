import pandas as pd
from pathlib import Path

def load_data(file):
    file = Path(file)

    if not file.exists():
        raise FileNotFoundError(f"Dataset not found: {file}")

    df = pd.read_csv(file)

    # Assume last column is target (defect label)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]

    return X, y, feature_names