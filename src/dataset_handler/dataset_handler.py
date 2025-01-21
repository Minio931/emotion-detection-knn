
from sklearn.model_selection import train_test_split

from src.audio_preprocessing import AudioFileFactory
from src.file_reader import FileManager


class DatasetHandler():
    def __init__(self, data):
        self.data = data

    def split_data(self, test_size=0.2, random_seed=None):
        return train_test_split(self.data, test_size=test_size, random_state=random_seed)

    def save_data_to_file(self, data, file_path):
        FileManager.save_to_csv(data, file_path)

    def assign_data_from_file(self, file_path, dtype=None):
        data = self.__load_data_from_file(file_path, dtype=dtype)
        if len(data) > 0:
            self.data = AudioFileFactory.create_audio_files_from_csv(data)

    def __load_data_from_file(self, file_path, dtype=None):
        return FileManager.read_from_csv(file_path, dtype=dtype)