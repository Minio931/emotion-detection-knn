from src.enums import NormalizeMethod
from .audio_preprocessing import AudioFileProcessor
from .dataset_handler import DatasetHandler
from .model_trainer import ModelTrainer


def main():
    audio_files_path = '/home/dominik/.cache/kagglehub/datasets/uwrfkaggler/ravdess-emotional-speech-audio/versions/1'
    audio_file_processor = AudioFileProcessor(
        audio_files_path,
        '.wav',
        '/home/dominik/Projects/emotion-detection-knn/visualizations')

    print("1. Preprocessowanie danych")
    audio_file_processor.load_audio_files()
    audio_file_processor.update_features_for_files()
    method = NormalizeMethod.STANDARD_SCALER
    audio_file_processor.normalize_features(method.value)
    audio_file_processor.visualize_features()

    print("2. Podział danych na zbiór treningowy i testowy")
    audio_files_data = audio_file_processor.audio_files
    dataset_handler = DatasetHandler(audio_files_data)
    train_data, test_data = dataset_handler.split_data()

    print(f"Zbiór treningowy: {len(train_data)}")
    print(f"Zbiór testowy: {len(test_data)}")

    print("3. Zapis danych do pliku")
    dataset_handler.save_data_to_file(train_data, '/home/dominik/Projects/emotion-detection-knn/data/train_data.csv')
    dataset_handler.save_data_to_file(test_data, '/home/dominik/Projects/emotion-detection-knn/data/test_data.csv')

    print("4. Uczenie modelu")
    knn_model = ModelTrainer(train_data, test_data, '/home/dominik/Projects/emotion-detection-knn/results')
    knn_model.train(5)

    print("5. Walidacja modelu")
    knn_model.validate()

if __name__ == "__main__":
    main()