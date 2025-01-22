from src.audio_preprocessing.audio_file import AudioFile

import numpy as np
import ast
import re

from src.decorators import logger


class AudioFileFactory:
    @logger(description="Tworzenie obiektów AudioFile")
    @staticmethod
    def create_audio_files(file_paths):
        audio_files = []
        for file_data in file_paths:
            audio_properties = AudioFileFactory.extract_audio_properties(file_data['file_name'])
            audio_files.append(AudioFile(file_data['path'], *audio_properties))

        return audio_files

    @logger(description="Tworzenie obiektów AudioFile z pliku CSV")
    @staticmethod
    def create_audio_files_from_csv(data):
        audio_files = []
        for index, row in data.iterrows():
                features = np.array(ast.literal_eval(re.sub(r'\s+', ', ', row['features'].strip())))
                audio_files.append(AudioFile(
                    row['path'],
                    row['modality'],
                    row['vocal_channel'],
                    row['emotion'],
                    row['emotional_intensity'],
                    row['statement'],
                    row['repetition'],
                    row['actor'],
                    features
                ))

        return audio_files

    @staticmethod
    def extract_audio_properties(file_name):
        return file_name.split('-')