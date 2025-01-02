import pandas as pd

from functions import create_directory_if_not_exists


def save_to_csv(file_path, data_frame):
    directory_path = './data'
    create_directory_if_not_exists(directory_path)

    path = directory_path + file_path
    data_frame.to_csv(path, index=False)
    print(f'Saved to {path}')
