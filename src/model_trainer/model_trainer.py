
import os

import numpy as np
import pandas as pd
from matplotlib.pyplot import xlabel

from tabulate import tabulate
from datetime import datetime

from src.decorators import logger

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from ..file_reader import FileManager
from ..visualization_manager import VisualizationManager


class ModelTrainer:
    def __init__(self, training_data, validation_data, model_folder_path="./results"):
        self.training_data = training_data
        self.validation_data = validation_data
        self.model_folder_path = model_folder_path
        self.model_code = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.knn_model = None
        self.predictions = None
        self.visualization_manager = VisualizationManager(model_folder_path + '/visualizations')

    @logger(description='Trenowanie modelu')
    def train(self, number_of_neighbours=5, data_columns=['features'], label_column='emotion'):
        training_data = self.__filter_columns(self.training_data, data_columns)
        training_labels = self.__filter_columns(self.training_data, label_column)

        self.knn_model = KNeighborsClassifier(n_neighbors=number_of_neighbours)
        self.knn_model.fit(training_data, training_labels)

    @logger(description='Walidacja modelu')
    def validate(self, data_columns=['features'], label_column='emotion'):
        validation_data = self.__filter_columns(self.validation_data, data_columns)
        validation_labels = self.__filter_columns(self.validation_data, label_column)

        self.predictions = self.knn_model.predict(validation_data)
        self.__generate_classification_report(self.predictions, validation_labels)

    @logger(description='Generowanie raportu')
    def __generate_classification_report(self, predictions, labels):
        report = classification_report(labels, predictions, output_dict=True)
        tabular_report = self.__generate_tabular_report(report)
        self.__save_report_to_file(tabular_report)
        self.__generate_confusion_matrix(predictions, labels)


    def __generate_confusion_matrix(self, predictions, labels):
        conf_matrix = confusion_matrix(labels, predictions)
        self.visualization_manager.visualize_confusion_matrix(
            conf_matrix,
            labels,
            plot_name='confusion_matrix',
            figsize=(10, 10),
            title='Confusion Matrix',
            xlabel='Predicted',
            ylabel='Actual'
        )

    def __generate_tabular_report(self, report):
        table_data = []
        for label, metrics in report.items():
            if label not in ['accuracy', 'macro avg', 'weighted avg']:
                table_data.append([
                    label,
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1-score'],
                    int(metrics['support'])
                ])
            elif label in ['accuracy']:
                table_data.append([label, '', '', metrics, ''])

        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]

        table = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f")
        return table

    def __save_report_to_file(self, report):
        report_path = os.path.join(self.model_folder_path + '/reports', f'classification_report_{self.model_code}.txt')
        FileManager.save_to_file(report_path, report, 'Classification Report')


    def __filter_columns(self, data, columns):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Podałeś zły typ danych. Oczekiwano DataFrame")

        return np.vstack([item[columns].values for item in data])

