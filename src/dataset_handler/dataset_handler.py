
from sklearn.model_selection import train_test_split

from src.audio_preprocessing import AudioFileFactory
from src.decorators import logger
from src.file_reader import FileManager


class DatasetHandler():
    def __init__(self, data):
        self.data = data

    @logger(description="Podział danych na zbiór treningowy i testowy")
    def split_data(self, test_size=0.2, random_seed=1337):
        return train_test_split(self.data, test_size=test_size, random_state=random_seed)

    @staticmethod
    @logger(description="Zapis danych do pliku")
    def save_data_to_file(data, file_path, file_name='data'):
        FileManager.save_to_csv(data, file_path, file_name)

    @logger(description="Wczytywanie danych z pliku")
    def assign_data_from_file(self, file_path, dtype=None):
        data = self.__load_data_from_file(file_path, dtype=dtype)
        if len(data) > 0:
            self.data = AudioFileFactory.create_audio_files_from_csv(data)

    def __load_data_from_file(self, file_path, dtype=None):
        return FileManager.read_from_csv(file_path, dtype=dtype)