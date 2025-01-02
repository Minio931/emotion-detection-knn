import librosa
import numpy as np

def extract_feature(audio_file):
    print(f'Extracting features for: {audio_file.path}')
    audio, sr = librosa.load(audio_file.path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=audio).T, axis=0)

    return np.hstack([mfccs, chroma, zcr, rms])