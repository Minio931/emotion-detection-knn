from emotion_detection_trainer import EmotionDetectionTrainer


# /home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1
def main():
    audio_files_path = '/home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
    emotion_detection_trainer = EmotionDetectionTrainer(audio_files_path)
    emotion_detection_trainer.initialize()
    emotion_detection_trainer.generate_plot_for_properties('emotion')
    emotion_detection_trainer.generate_plot_for_properties('emotional_intensity')
    emotion_detection_trainer.generate_plot_for_properties('statement')
    emotion_detection_trainer.generate_plot_for_properties('repetition')
    emotion_detection_trainer.generate_plot_for_properties('actor')
    emotion_detection_trainer.generate_plot_for_properties('vocal_channel')
    emotion_detection_trainer.generate_plot_for_properties('modality')




if __name__ == "__main__":
    main()