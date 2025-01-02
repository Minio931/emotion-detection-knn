import ast
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

from functions import extract_feature, save_to_csv, create_directory_if_not_exists
from functions.read_from_csv import read_from_csv
from models.audio_file import AudioFile





class EmotionDetectionTrainer:
    def __init__(self, audio_folder):
        self.audio_folder = audio_folder
        self.audio_files = []
        self.training_data = []
        self.validation_data = []

    def initialize(self):
        self.read_data_from_file()
        if len(self.audio_files) == 0:
            self.__load_audio_files()
            self.__extract_features()
            self.__normalize_features()
            self.save_data_to_file()

    def train(self, k=5):
        self.__split_data()
        self.__plot_data(self.training_data, 'training_data')
        self.__plot_data(self.validation_data, 'validation_data')
        self.train_knn(k)

    def generate_plot_for_properties(self, property_name):
        data = pd.DataFrame([audio_file.__dict__() for audio_file in self.audio_files])
        self.__visualize_data(data, property_name)

    def save_data_to_file(self):
        data = pd.DataFrame([audio_file.__dict_original__() for audio_file in self.audio_files])
        save_to_csv('/audio_files.csv', data)

    def read_data_from_file(self):
        data = read_from_csv('/audio_files.csv', dtype={'modality': str, 'vocal_channel': str, 'emotion': str, 'emotional_intensity': str, 'statement': str, 'repetition': str, 'actor': str})
        if len(data) > 0:
            for index, row in data.iterrows():
                features = np.array(ast.literal_eval(re.sub(r'\s+', ', ', row['features'].strip())))
                # Initialize the AudioFile object
                audio_file = AudioFile(
                    path=row['path'],
                    modality=row['modality'],
                    vocal_channel=row['vocal_channel'],
                    emotion=row['emotion'],
                    emotional_intensity=row['emotional_intensity'],
                    statement=row['statement'],
                    repetition=row['repetition'],
                    actor=row['actor'],
                    features=features
                )
                self.audio_files.append(audio_file)

    def __visualize_data(self, data, property_name):
        sns.countplot(data[property_name])
        plt.title(f'Distribution of {property_name}')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        create_directory_if_not_exists('./plots')
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
            features = extract_feature(audio_file)
            audio_file.features = features

    def __normalize_features(self):
        for audio_file in self.audio_files:
            audio_file.normalize_features()


    def __split_data(self):
        number_of_training_samples = int(len(self.audio_files) * 0.8)
        number_of_validation_samples = len(self.audio_files) - number_of_training_samples
        self.training_data = [audio_file for audio_file in self.audio_files[:number_of_training_samples]]
        self.validation_data = [audio_file for audio_file in self.audio_files[number_of_training_samples:]]

        save_to_csv('/training_data.csv', pd.DataFrame([audio_file.__dict_original__() for audio_file in self.training_data]))
        save_to_csv('/validation_data.csv', pd.DataFrame([audio_file.__dict_original__() for audio_file in self.validation_data]))

    def __plot_data(self, audio_files, plot_name='features'):
        data = pd.DataFrame([audio_file.__dict__() for audio_file in audio_files])

        tsne = TSNE(n_components=2, perplexity=50, random_state=42)

        feature_matrix = np.vstack(data['features'].tolist())
        reduced_features = tsne.fit_transform(feature_matrix)

        data['TSNE_1'] = reduced_features[:, 0]
        data['TSNE_2'] = reduced_features[:, 1]

        plt.figure(figsize=(10, 10))
        for emotion in data['emotion'].unique():
            subset = data[data['emotion'] == emotion]
            plt.scatter(subset['TSNE_1'], subset['TSNE_2'], label=emotion, alpha=0.5)

        plt.title('t-SNE of Features')
        plt.xlabel('TSNE_1')
        plt.ylabel('TSNE_2')
        plt.legend()
        plt.grid()
        plt.savefig(f'./plots/{plot_name}_tsne.png', dpi=300, bbox_inches='tight')
        plt.clf()


    def train_knn(self, k=5):
        training_features = np.vstack([audio_file.features for audio_file in self.training_data])
        training_labels = [audio_file.emotion for audio_file in self.training_data]

        validation_features = np.vstack([audio_file.features for audio_file in self.validation_data])
        validation_labels = [audio_file.emotion for audio_file in self.validation_data]

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(training_features, training_labels)

        predictions = knn.predict(validation_features)

        print(f'Accuracy: {np.mean(predictions == validation_labels)}')
        print(f"Classification Report: {classification_report(validation_labels, predictions)}")
        print(f"Confusion Matrix: {pd.crosstab(np.array(validation_labels), predictions, rownames=['Actual'], colnames=['Predicted'])}")








