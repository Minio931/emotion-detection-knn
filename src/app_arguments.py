import argparse
import pandas as pd
import os

from dotenv import load_dotenv, dotenv_values
from src import FileManager
from src.enums import NormalizeMethod
from src.audio_preprocessing import AudioFileProcessor
from src.dataset_handler import DatasetHandler
from src.model_trainer import ModelTrainer

def parse_arguments():
    parser = argparse.ArgumentParser(description="Konfiguracja aplikacji do detekcji emocji")
    parser.add_argument("--normalization", type=str, choices=["standard", "minmax", "l2", "log", "none"],
                        default="none", help="Metoda normalizacji cech")
    parser.add_argument("--neighbors", type=int, default=5, help="Liczba sąsiadów KNN")
    parser.add_argument("--visualize", action="store_true", help="Czy generować wizualizacje")
    parser.add_argument("--train_split", type=float, default=0.8, help="Procent danych na zbiór treningowy")
    return parser.parse_args()

def main():
    args = parse_arguments()

    # Konfiguracja ścieżek
    config = dotenv_values('.env')
    dir_path = config['DIR_PATH']
    audio_files_path = config['AUDIO_FILES_PATH']
    data_path = os.path.join(dir_path, 'data')
    visualizations_path = os.path.join(dir_path, 'visualizations')
    model_path = os.path.join(dir_path, 'model')

    # Procesor plików audio
    audio_file_processor = AudioFileProcessor(
        audio_files_path,
        '.wav',
        os.path.join(dir_path, 'visualizations'))

    print("1. Preprocessowanie danych")
    if not FileManager.file_exists(os.path.join(data_path, 'audio_files.csv')):
        audio_file_processor.load_audio_files()
        audio_file_processor.update_features_for_files()
        audio_files = audio_file_processor.audio_files
        data_frame = AudioFileProcessor.format_audio_files_as_dataframe(audio_files)
        FileManager.save_to_csv(data_frame, os.path.join(data_path, 'audio_files.csv'))
    else:
        audio_file_processor.load_audio_files_from_csv(os.path.join(data_path, 'audio_files.csv'))
        audio_files = audio_file_processor.audio_files

    # Normalizacja
    method = args.normalization
    if method != "none":
        normalize_method = {
            "standard": NormalizeMethod.STANDARD_SCALER.value,
            "minmax": NormalizeMethod.MIN_MAX_SCALER.value,
            "l2": NormalizeMethod.L2_NORMALIZER.value,
            "log": NormalizeMethod.LOGARITHMIC_SCALER.value
        }[method]
        print(f"Normalizacja cech za pomocą metody: {method}")
        audio_file_processor.normalize_features(normalize_method)
    else:
        print("Brak normalizacji cech")

    # Wizualizacja cech
    if args.visualize:
        print("3. Wizualizacja cech")
        audio_file_processor.visualize_features(plot_name=f'pca_features_{method}')
        audio_file_processor.visualize_properties_distribution(properties=['emotion'], plot_name='emotion_distribution')

    # Podział danych na trening i test
    print("4. Podział danych na zbiór treningowy i testowy")
    dataset_handler = DatasetHandler(audio_files)
    train_data, test_data = dataset_handler.split_data(test_size=(1 - args.train_split))

    print(f"Zbiór treningowy: {len(train_data)}")
    print(f"Zbiór testowy: {len(test_data)}")

    # Zapis danych
    print("5. Zapis danych do pliku")
    pd_frame_training_data = AudioFileProcessor.format_audio_files_as_dataframe(train_data)
    pd_frame_test_data = AudioFileProcessor.format_audio_files_as_dataframe(test_data)
    DatasetHandler.save_data_to_file(pd_frame_training_data, os.path.join(data_path), 'train_data.csv')
    DatasetHandler.save_data_to_file(pd_frame_test_data, os.path.join(data_path), 'test_data.csv')

    # Trening modelu
    print("6. Uczenie modelu")
    knn_model = ModelTrainer(pd_frame_training_data, pd_frame_test_data, os.path.join(model_path))
    knn_model.train(args.neighbors)

    # Walidacja modelu
    print("7. Walidacja modelu")
    knn_model.validate()

    print("Zakończono")

if __name__ == "__main__":
    main()
