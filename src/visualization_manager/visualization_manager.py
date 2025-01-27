import os
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
        output_folder = kwargs.get('output_folder', './plots')

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(properties, list):
            raise TypeError("properties must be a list of column names")

        # Validate columns
        missing_columns = [col for col in properties if col not in data.columns]
        if missing_columns:
            raise ValueError(f"The following properties are missing in the DataFrame: {missing_columns}")

        # Plot each property separately
        plt.figure(figsize=(10, 10))
        for property_name in properties:
            if data[property_name].isnull().all():
                print(f"Property '{property_name}' has no data to plot. Skipping.")
                continue

            sns.countplot(data=data, x=property_name)
            plt.title(f'Distribution of {property_name}')
            plt.xlabel(property_name)
            plt.ylabel('Count')

        propeties_names = ', '.join(properties)
        self.__save_plot(kwargs.get('plot_name', f'properties_distribution_{propeties_names}'))
        self.__clear_plot()

    @logger(description="Wizualizacja macierzy korelacji")
    def visualize_confusion_matrix(self, data, labels, **kwargs):
        sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f",
                    xticklabels=[f"{i}" for i in set(labels)],
                    yticklabels=[f"{i}" for i in set(labels)])

        plt.title(kwargs.get('title', 'Correlation Matrix'))
        plt.xlabel(kwargs.get('xlabel', 'Predicted'))
        plt.ylabel(kwargs.get('ylabel', 'Actual'))
        plt.xticks(rotation=kwargs.get('rotation', 0))
        plt.grid(False)

        self.__save_plot(kwargs.get('plot_name', 'correlation_matrix'))
        self.__clear_plot()

    @logger(description="Wizualizacja rozkładu klas")
    def visualize_pca_features(self, audio_files, **kwargs):
        data = pd.DataFrame([audio_file.__dict__() for audio_file in audio_files])

        if 'features' not in data.columns:
            raise ValueError("Features column is missing in the data.")

        def handle_missing_features(features):
            features = np.array(features)
            if np.isnan(features).any():
                nan_mean = np.nanmean(features)
                features = np.nan_to_num(features, nan=nan_mean)
            return features

        data['features'] = data['features'].apply(handle_missing_features)

        features_array = np.vstack(data['features'])

        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(features_array)

        self.__configure_plot(**kwargs)

        for emotion in data['emotion'].unique():
            indices = data['emotion'] == emotion
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
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Ensure directory exists
            if not os.path.exists(self.visualizations_folder_path):
                os.makedirs(self.visualizations_folder_path)

            # Save the plot
            file_path = os.path.join(self.visualizations_folder_path, f'{plot_name}_{timestamp}.png')
            plt.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {file_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")

    def __clear_plot(self):
       plt.clf()