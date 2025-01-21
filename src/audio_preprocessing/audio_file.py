from enums import *
import numpy as np
from sklearn.preprocessing import StandardScaler


class AudioFile:
    def __init__(self, path, modality, vocal_channel, emotion, emotional_intensity, statement, repetition, actor, features=None):
        self.path = path
        self.modality = str(modality)
        self.vocal_channel = str(vocal_channel)
        self.emotion = str(emotion)
        self.emotional_intensity = str(emotional_intensity)
        self.statement = str(statement)
        self.repetition = str(repetition)
        self.actor = str(actor)
        self.features = features

    def __str__(self):
        return f'Path: {self.path}\n' \
               f'Modality: {self.modality}\n' \
               f'Vocal Channel: {self.vocal_channel}\n' \
               f'Emotion: {self.emotion}\n' \
               f'Emotional Intensity: {self.emotional_intensity}\n' \
               f'Statement: {self.statement}\n' \
               f'Repetition: {self.repetition}\n' \
               f'Actor: {self.actor}\n' \
               f'Features: {self.features}'

    def __repr__(self):
        return self.__str__()

    def __dict__(self):
        return {
            'path': self.path,
            'modality': Modality.get_enum_value(self.modality),
            'vocal_channel': VocalChannel.get_enum_value(self.vocal_channel),
            'emotion': Emotion.get_enum_value(self.emotion),
            'emotional_intensity': EmotionalIntensity.get_enum_value(self.emotional_intensity),
            'statement': Statement.get_enum_value(self.statement),
            'repetition': Repetition.get_enum_value(self.repetition),
            'actor': Actor.get_enum_value(int(self.actor) % 2),
            'features': self.features
        }

    def __dict_original__(self):
        return {
            'path': self.path,
            'modality': self.modality,
            'vocal_channel': self.vocal_channel,
            'emotion': self.emotion,
            'emotional_intensity': self.emotional_intensity,
            'statement': self.statement,
            'repetition': self.repetition,
            'actor': self.actor,
            'features': self.features
        }


    def normalize_features(self):
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features.reshape(-1, 1)).flatten()

        mean = np.mean(self.features)
        std = np.std(self.features)

        print(f'Mean: {mean}, Std: {std}')



