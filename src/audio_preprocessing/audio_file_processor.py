import sys
import librosa
import numpy as np

from src.enums import NormalizeMethod
from src.audio_preprocessing.audio_file_factory import AudioFileFactory
from src.audio_preprocessing.normalization_handler import NormalizationHandler
from src.file_reader import FileManager
from src.decorators import logger
from src.visualization_manager.visualization_manager import VisualizationManager


class AudioFileProcessor:
    def __init__(self, audio_folder_path, extension, visulaizations_folder_path="../visualizations"):
        self.audio_folder_path = audio_folder_path
        self.extension = extension
        self.audio_files = []
        self.visualization_manager = VisualizationManager(visualizations_folder_path=visulaizations_folder_path)

    @logger(description="Wczytywanie plików audio")
    def load_audio_files(self):
        audio_names_with_paths = self.__get_audio_file_paths()
        self.__update_audio_files(AudioFileFactory.create_audio_files(audio_names_with_paths))

    @logger(description="Aktualizacja cech dla plików")
    def update_features_for_files(self):
        for audio_file in self.audio_files:
            features = self.__extract_file_features(audio_file)
            audio_file = self.__update_audio_file_features(audio_file, features)

    @logger(description="Normalizacja cech")
    def normalize_features(self, method):
        self.__normalize_features_with_selected_method(method)

    @logger(description="Wizualizacja cech")
    def visualize_features(self, **kwargs):
        self.visualization_manager.visualize_pca_features(self.audio_files, **kwargs)

    @logger(description="Wizualizacja propercji")
    def visualize_properties_distribution(self, **kwargs):
        self.visualization_manager.visualize_properties_distribution(self.audio_files, **kwargs)


    def __update_audio_files(self, audio_files):
        self.audio_files = audio_files

    def __get_audio_file_paths(self):
        try:
            audio_name_with_paths = FileManager.get_files_with_paths(self.audio_folder_path, self.extension)
            if len(audio_name_with_paths) <= 0:
                raise FileNotFoundError(f"Nie znaleziono żadnych plików z rozszerzeniem {self.extension} w folderze {self.audio_folder_path}.")

            return audio_name_with_paths
        except FileNotFoundError as error:
            print(f"Błąd: {error}")
            sys.exit(1)
        except Exception as e:
            print(f"Nieoczekiwany błąd: {e}")


    def __extract_file_features(self, audio_file):
        print(f"Ekstrakcja cech dla pliku: {audio_file.path}")
        audio, sr = librosa.load(audio_file.path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

        return np.hstack([mfccs, chroma, zcr, rms])

    def __update_audio_file_features(self, audio_file, features):
        audio_file.features = features
        return audio_file

    def __normalize_features_with_selected_method(self, method):
        for audio_file in self.audio_files:
            normalization_method = NormalizationHandler.get_method(method)
            normalized_features = normalization_method(audio_file.features)
            audio_file = self.__update_audio_file_features(audio_file, normalized_features)


