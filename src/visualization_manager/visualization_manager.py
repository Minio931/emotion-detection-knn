import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from datetime import datetime
import seaborn as sns
from sklearn.decomposition import PCA

from src.decorators import logger
from src.file_reader import FileManager


class VisualizationManager:
    def __init__(self, visualizations_folder_path):
        self.visualizations_folder_path = visualizations_folder_path

    @logger(description="Wizualizacja rozkładu cech")
    def visualize_properties_distribution(self, data, **kwargs):
        properties = kwargs.get('properties', [])

        if not properties:
            print("Brak właściwości do wizualizacji")
            return

        data_long = data[properties].melt(var_name="Property", value_name="Value")
        sns.barplot(data=data_long, x="Property", y="Value", dodge=False)

        self.__configure_plot(**kwargs)
        self.__save_plot(kwargs.get('plot_name', 'properties_distribution'))
        self.__clear_plot()

    @logger(description="Wizualizacja macierzy korelacji")
    def visualize_confusion_matrix(self, data, labels, **kwargs):
        sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f",
                    xticklabels = [f"Predicted {label}" for label in labels],
                    yticklabels = [f"Actual {label}" for label in labels])

        self.__configure_plot(**kwargs)
        self.__save_plot(kwargs.get('plot_name', 'correlation_matrix'))
        self.__clear_plot()

    @logger(description="Wizualizacja rozkładu klas")
    def visualize_pca_features(self, audio_files, **kwargs):
        data = pd.DataFrame([audio_file.__dict__() for audio_file in audio_files])

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(data['features'].to_list())

        self.__configure_plot(**kwargs)

        for emotion in data['emotion'].unique():
            indices = data['emotion'] == emotion  # Wybór indeksów dla konkretnej emocji
            plt.scatter(reduced_features[indices, 0], reduced_features[indices, 1], label=emotion, alpha=0.7)

        self.__save_plot(kwargs.get('plot_name', 'features_pca'))
        self.__clear_plot()


    def __configure_plot(self, **kwargs):
        plt.figure(figsize=kwargs.get('figsize', (10, 10)))
        plt.title(kwargs.get('title', 'Tytuł'))
        plt.xlabel(kwargs.get('xlabel', 'X'))
        plt.ylabel(kwargs.get('ylabel', 'Y'))
        plt.legend(title=kwargs.get('legend_title', 'Legend'))
        plt.xticks(rotation=kwargs.get('rotation', 0))
        plt.grid()

    def __save_plot(self, plot_name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        FileManager.create_directory_if_not_exists(self.visualizations_folder_path)
        plt.savefig(f'{self.visualizations_folder_path}/{plot_name}_{timestamp}.png',
                    dpi=300,
                    bbox_inches='tight')

    def __clear_plot(self):
       plt.clf()