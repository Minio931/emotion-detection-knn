from enums import *


class AudioFile:
    def __init__(self, path, modality, vocal_channel, emotion, emotional_intensity, statement, repetition, actor):
        self.path = path
        self.modality = Modality.get_enum_value(modality)
        self.vocal_channel = VocalChannel.get_enum_value(vocal_channel)
        self.emotion = Emotion.get_enum_value(emotion)
        self.emotional_intensity = EmotionalIntensity.get_enum_value(emotional_intensity)
        self.statement = Statement.get_enum_value(statement)
        self.repetition = Repetition.get_enum_value(repetition)
        self.actor = Actor.get_enum_value(int(actor) % 2)
        self.features = []

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
            'modality': self.modality,
            'vocal_channel': self.vocal_channel,
            'emotion': self.emotion,
            'emotional_intensity': self.emotional_intensity,
            'statement': self.statement,
            'repetition': self.repetition,
            'actor': self.actor,
            'features': self.features
        }


