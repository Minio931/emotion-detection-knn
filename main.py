from emotion_detection_trainer import EmotionDetectionTrainer
from src.audio_preprocessing.audio_file_processor import AudioFileProcessor


# /home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1
def main():
    audio_files_path = '/home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
    # emotion_detection_trainer = EmotionDetectionTrainer(audio_files_path)
    # emotion_detection_trainer.initialize()
    # emotion_detection_trainer.train(5)
    # print(FileReader.get_files_with_paths('/home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1', '.wav'))
    audio_file_processor = AudioFileProcessor(audio_files_path)
    audio_file_processor.load_audio_files()


if __name__ == "__main__":
    main()