import pandas as pd

def load_data(file):
    """
    file: file path (string) OR Streamlit UploadedFile
    """

    if hasattr(file, "read"):
        # Uploaded file
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y