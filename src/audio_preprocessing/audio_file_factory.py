from src.audio_preprocessing.audio_file import AudioFile

import numpy as np
import ast
import re

from src.decorators import logger
from src.enums import Modality, VocalChannel, Emotion, EmotionalIntensity, Statement, Repetition, Actor


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
                Modality[row['modality'].upper()].value,
                VocalChannel[row['vocal_channel'].upper()].value,
                Emotion[row['emotion'].upper()].value,
                EmotionalIntensity[row['emotional_intensity'].upper()].value,
                Statement[row['statement'].upper()].value,
                Repetition[row['repetition'].upper()].value,
                Actor[row['actor'].upper()].value if row['actor'].upper() in Actor.__members__ else int(row['actor']),
                features
            ))
        return audio_files

    @staticmethod
    def extract_audio_properties(file_name):
        return file_name.split('-')