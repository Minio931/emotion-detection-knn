import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from datetime import datetime
import seaborn as sns

from src.file_reader import FileManager


class VisualizationManager:
    def __init__(self, visualizations_folder_path):
        self.visualizations_folder_path = visualizations_folder_path

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

    def visualize_correlation_matrix(self, data, **kwargs):
        corr = data.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")

        self.__configure_plot(**kwargs)
        self.__save_plot(kwargs.get('plot_name', 'correlation_matrix'))
        self.__clear_plot()

    def visualize_tsne_features(self, audio_files, **kwargs):
        data = pd.DataFrame([audio_file.__dict__() for audio_file in audio_files])

        tsne = TSNE(n_components=2, perplexity=50, random_state=42)

        feature_matrix = np.vstack(data['features'].tolist())
        reduced_features = tsne.fit_transform(feature_matrix)

        data['TSNE_1'] = reduced_features[:, 0]
        data['TSNE_2'] = reduced_features[:, 1]

        plt.figure(figsize=kwargs.get('figsize', (10, 10)))

        for emotion in data['emotion'].unique():
            subset = data[data['emotion'] == emotion]
            plt.scatter(subset['TSNE_1'], subset['TSNE_2'], label=emotion, alpha=0.5)

        self.__configure_plot(**kwargs)
        self.__save_plot(kwargs.get('plot_name', 'features_tsne'))
        self.__clear_plot()


    def __configure_plot(self, **kwargs):
        plt.title(kwargs.get('title', 'Tytuł'))
        plt.xlabel(kwargs.get('xlabel', 'X'))
        plt.ylabel(kwargs.get('ylabel', 'Y'))
        plt.legend(title=kwargs.get('legend_title', 'Legend'))
        plt.xticks(rotation=kwargs.get('rotation', 0))
        plt.figure(figsize=kwargs.get('figsize', (10, 10)))
        plt.grid()

    def __save_plot(self, plot_name):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        FileManager.create_directory_if_not_exists(self.visualizations_folder_path)
        plt.savefig(f'{self.visualizations_folder_path}/{plot_name}_{timestamp}.png', dpi=300, bbox_inches='tight')

   def __clear_plot(self):
       plt.clf()