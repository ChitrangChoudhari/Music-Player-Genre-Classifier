"""
Trains a genre classifier and saves it to genre_model.pkl.

Usage:
    1. Download a labeled dataset, e.g. GTZAN:
       https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
    2. Point DATASET_DIR at the folder that contains one subfolder per genre,
       e.g. gtzan/genres_original/blues/*.wav, .../rock/*.wav, etc.
    3. Run: python train_genre_classifier.py
    4. Copy the resulting genre_model.pkl next to Test.py.
"""
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from genre_features import extract_features, GENRES

DATASET_DIR = "gtzan/genres_original"  # <-- change to your dataset path


def build_dataset():
    X, y = [], []
    for genre in GENRES:
        genre_dir = os.path.join(DATASET_DIR, genre)
        if not os.path.isdir(genre_dir):
            print(f"Skipping missing genre folder: {genre_dir}")
            continue
        for fname in os.listdir(genre_dir):
            if fname.lower().endswith((".wav", ".mp3", ".au")):
                path = os.path.join(genre_dir, fname)
                try:
                    X.append(extract_features(path))
                    y.append(genre)
                    print(f"Processed {path}")
                except Exception as e:
                    print(f"Skipping {path}: {e}")
    return np.array(X), np.array(y)


def main():
    print("Extracting features (this can take a while)...")
    X, y = build_dataset()

    if len(X) == 0:
        raise RuntimeError(
            f"No audio files found under {DATASET_DIR}. "
            "Update DATASET_DIR to point at your dataset."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    print("\nTest accuracy:", clf.score(X_test, y_test))
    print(classification_report(y_test, clf.predict(X_test)))

    joblib.dump({"model": clf, "scaler": scaler}, "genre_model.pkl")
    print("\nSaved trained model to genre_model.pkl")


if __name__ == "__main__":
    main()
