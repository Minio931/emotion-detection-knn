import os
import sys

import pandas as pd

class FileManager:

    @staticmethod
    def get_files_with_paths(folder_path, extension):
        return [
            {
                'path': os.path.join(root, file),
                'file_name':file.replace(extension, '')
            }
            for root, dirs, files in os.walk(folder_path)
            for file in files if file.endswith(extension)
        ]

    @staticmethod
    def save_to_csv(file_path, data_frame):
        if not isinstance(data_frame, pd.DataFrame):
            raise ValueError("Podałeś zły typ danych. Oczekiwano DataFrame")

        FileManager.create_directory_if_not_exists(file_path)
        data_frame.to_csv(file_path, index=False)
        print(f"Dane zostały zapisane do pliku: {file_path}")

    @staticmethod
    def read_from_csv(file_path, **kwargs):
        try:
            return pd.read_csv(file_path, **kwargs)
        except FileNotFoundError:
            print(f'Nie znaleziono pliku: {file_path}')
            sys.exit(1)


    @staticmethod
    def create_directory_if_not_exists(file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
            print(f'Created directory: {file_path}')
        else:
            print(f'Directory already exists: {file_path}')