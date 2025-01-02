from emotion_detection_trainer import EmotionDetectionTrainer


# /home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1
def main():
    audio_files_path = '/home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
    emotion_detection_trainer = EmotionDetectionTrainer(audio_files_path)
    emotion_detection_trainer.initialize()
    emotion_detection_trainer.train(5)





if __name__ == "__main__":
    main()