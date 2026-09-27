"""
Shared audio feature extraction for genre classification.
Used by both train_genre_classifier.py (training) and Player.py (inference).
"""
import numpy as np
import librosa

# Standard GTZAN genre set — change this list if you train on different labels
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]


def extract_features(file_path, duration=30):
    """
    Load up to `duration` seconds of audio and return a fixed-length
    feature vector summarizing timbre, harmony, and rhythm.
    """
    y, sr = librosa.load(file_path, duration=duration)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    features = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        np.mean(spec_centroid), np.std(spec_centroid),
        np.mean(spec_bw), np.std(spec_bw),
        np.mean(rolloff), np.std(rolloff),
        np.mean(zcr), np.std(zcr),
        tempo,
    ])
    return features
