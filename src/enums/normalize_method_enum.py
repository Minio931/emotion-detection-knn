from enum import Enum

class NormalizeMethod(Enum):
    STANDARD_SCALER = 'StandardScaler',
    MIN_MAX_SCALER = 'MinMaxScaler',
    L2_NORMALIZER = 'L2Normalizer',
    LOGARITHMIC_SCALER = 'LogarithmicScaler',