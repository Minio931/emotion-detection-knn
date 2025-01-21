import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from enums import NormalizeMethod


class NormalizationHandler:
    @staticmethod
    def standard_scaler(data):
        scaler = StandardScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()

    @staticmethod
    def min_max_scaling(data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data.reshape(-1, 1)).flatten()

    @staticmethod
    def l2_normalizer(data):
        normalizer = Normalizer(norm='l2')
        return normalizer.transform(data.reshape(-1, 1)).flatten()

    @staticmethod
    def logarithmic_scaler(data):
        return np.log1p(data)

    @staticmethod
    def get_method(method):
        match method:
            case NormalizeMethod.STANDARD_SCALER:
                return NormalizationHandler.standard_scaler
            case NormalizeMethod.MIN_MAX_SCALER:
                return NormalizationHandler.min_max_scaling
            case NormalizeMethod.L2_NORMALIZER:
                return NormalizationHandler.l2_normalizer
            case NormalizeMethod.LOGARITHMIC_SCALER:
                return NormalizationHandler.logarithmic_scaler
            case _:
                raise ValueError(f'Normalization method {method} not found')

