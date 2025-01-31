import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from enums import NormalizeMethod
from src.decorators import logger


class NormalizationHandler:

    @logger(description='Normalizacja danych za pomocą StandardScaler')
    @staticmethod
    def standard_scaler(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data).flatten()

    @logger(description='Normalizacja danych za pomocą MinMaxScaler')
    @staticmethod
    def min_max_scaling(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data).flatten()

    @logger(description='Normalizacja danych za pomocą Normalizer')
    @staticmethod
    def l2_normalizer(data):
        normalizer = Normalizer(norm='l2')
        return normalizer.transform(data).flatten()

    @logger(description='Normalizacja danych za pomocą logarytmu')
    @staticmethod
    def logarithmic_scaler(data):
        return np.log1p(data)

    @staticmethod
    def get_method(method):
        if method == NormalizeMethod.STANDARD_SCALER.value:
            return NormalizationHandler.standard_scaler
        elif method == NormalizeMethod.MIN_MAX_SCALER.value:
            return NormalizationHandler.min_max_scaling
        elif method == NormalizeMethod.L2_NORMALIZER.value:
            return NormalizationHandler.l2_normalizer
        elif method == NormalizeMethod.LOGARITHMIC_SCALER.value:
            return NormalizationHandler.logarithmic_scaler
        else:
            raise ValueError(
                f"Nie znaleziono metody: {method}, dostępne opcje: {[m.value for m in NormalizeMethod]}")




