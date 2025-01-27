import pandas as pd
import os

from dotenv import load_dotenv, dotenv_values

from src import FileManager
from src.enums import NormalizeMethod
from .audio_preprocessing import AudioFileProcessor
from .dataset_handler import DatasetHandler
from .model_trainer import ModelTrainer


def main():
    config = dotenv_values('.env')
    dir_path = config['DIR_PATH']
    audio_files_path = config['AUDIO_FILES_PATH']
    data_path = os.path.join(dir_path, 'data')
    visualizations_path = os.path.join(dir_path, 'visualizations')
    model_path = os.path.join(dir_path, 'model')

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


    correct_answer = False

    while not correct_answer:
        print("""
        2. Wybierz metodę normalizacji cech:
            1. Standard Scaler
            2. MinMax Scaler
            3. L2 Normalizer
            4. None
        """)
        normalization_method_input = input("Wybierz metodę normalizacji cech: ")
        method = ""
        match normalization_method_input:
            case '1':
                method = NormalizeMethod.STANDARD_SCALER.value
                audio_file_processor.normalize_features(method)
                correct_answer = True
            case '2':
                method = NormalizeMethod.MIN_MAX_SCALER.value
                audio_file_processor.normalize_features(method)
                correct_answer = True
            case '3':
                method = NormalizeMethod.L2_NORMALIZER.value
                audio_file_processor.normalize_features(method)
                correct_answer = True
            case '4':
                print("Brak normalizacji")
                correct_answer = True
            case _:
                print("Nieprawidłowy wpis")

    print("3. Wizualizacja cech")
    audio_file_processor.visualize_features(plot_name=f'pca_features_{method}')

    visualize_properties = int(input("Czy wizualizować cechy (1=tak, 0=nie): "))
    if visualize_properties == 1:
        audio_file_processor.visualize_properties_distribution(properties=['emotion'], plot_name='emotion_distribution')
        audio_file_processor.visualize_properties_distribution(properties=['modality'], plot_name='modality_distribution')
        audio_file_processor.visualize_properties_distribution(properties=['vocal_channel'], plot_name='vocal_channel_distribution')
        audio_file_processor.visualize_properties_distribution(properties=['emotional_intensity'], plot_name='emotional_intensity_distribution')
        audio_file_processor.visualize_properties_distribution(properties=['statement'], plot_name='statement_distribution')
        audio_file_processor.visualize_properties_distribution(properties=['repetition'], plot_name='repetition_distribution')
        audio_file_processor.visualize_properties_distribution(properties=['actor'], plot_name='actor_distribution')

    print("4. Podział danych na zbiór treningowy i testowy")
    dataset_handler = DatasetHandler(audio_files)
    correct_answer = False

    while not correct_answer:
        split_size = float(input("Podaj rozmiar zbioru treningowego (domyślnie=0.8): "))
        if split_size == 0:
            split_size = 0.8

        if split_size > 1 or split_size < 0:
            print("Nieprawidłowa wartość")
        else:
            correct_answer = True


    train_data, test_data = dataset_handler.split_data(test_size=(1 - split_size))

    print(f"Zbiór treningowy: {len(train_data)}")
    print(f"Zbiór testowy: {len(test_data)}")

    print("5. Zapis danych do pliku")
    pd_frame_training_data = AudioFileProcessor.format_audio_files_as_dataframe(train_data)
    pd_frame_test_data = AudioFileProcessor.format_audio_files_as_dataframe(test_data)
    DatasetHandler.save_data_to_file(pd_frame_training_data, os.path.join(data_path), 'train_data.csv')
    DatasetHandler.save_data_to_file(pd_frame_test_data, os.path.join(data_path), 'test_data.csv')

    print("6. Uczenie modelu")
    knn_model = ModelTrainer(pd_frame_training_data, pd_frame_test_data, os.path.join(model_path))

    number_of_neighbors = int(input("Podaj liczbę sąsiadów (domyślnie=5): "))
    if number_of_neighbors == 0:
        number_of_neighbors = 5

    knn_model.train(number_of_neighbors)

    print("7. Walidacja modelu")
    knn_model.validate()

    print("Zakończono")








if __name__ == "__main__":
    main()