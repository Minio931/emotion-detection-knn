import os
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


from models.audio_file import AudioFile


class EmotionDetectionTrainer:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder
        self.audio_files = []

    def initialize(self):
        self.__load_audio_files()
        self.__extract_features()

    def generate_plot_for_properties(self, property_name):
        data = pd.DataFrame([audio_file.__dict__() for audio_file in self.audio_files])
        self.__visualize_data(data, property_name)

    def __visualize_data(self, data, property_name):
        sns.countplot(data[property_name])
        plt.title(f'Distribution of {property_name}')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'./plots/{property_name}_distribution_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.clf()

    def __load_audio_files(self):
        for root, dirs, files in os.walk(self.audio_folder):
            for file in files:
                if file.endswith('.wav'):
                    print(f'Loading file: {file}')
                    file_path = os.path.join(root, file)
                    audio_properties = file.split('-')
                    audio_properties[len(audio_properties) - 1] = audio_properties[len(audio_properties) - 1].replace('.wav', '')
                    self.audio_files.append(AudioFile(file_path, *audio_properties))

    def __extract_features(self):
        for audio_file in self.audio_files:
            features = self.__extract_feature(audio_file)
            audio_file.features = features
            print(f'Audio File: {audio_file.path} ')


    def __extract_feature(self, audio_file):
        audio, sr = librosa.load(audio_file.path, sr=None)
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

        return np.hstack([mfccs, chroma, zcr, rms])



