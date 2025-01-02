import pandas as pd

def read_from_csv(file_path, dtype=None):
    path = './data' + file_path
    try:
        return pd.read_csv(path, dtype=dtype)
    except FileNotFoundError:
        return []