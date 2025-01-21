from enums import BaseEnum

class NormalizeMethod(BaseEnum):
    STANDARD_SCALER = 'StandardScaler',
    MIN_MAX_SCALER = 'MinMaxScaler',
    L2_NORMALIZER = 'L2Normalizer',
    LOGARITHMIC_SCALER = 'LogarithmicScaler',