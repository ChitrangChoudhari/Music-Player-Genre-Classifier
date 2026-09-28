"""
Trains a genre classifier and saves it to genre_model.pkl.

Supports merging MULTIPLE datasets (e.g. GTZAN for Western genres +
the Indian Music Genre dataset for Bollypop/Sufi/Ghazal/Carnatic/etc.)
without copying files into one shared folder.

Usage:
    1. Download each dataset separately and leave its folder structure as-is
       (one subfolder per genre inside each dataset's root).
       - GTZAN:      https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
       - IMG (Indian): https://www.kaggle.com/winchester19/indian-music-genre-dataset
    2. Edit the DATASETS list below: for each dataset, set its root path and
       a mapping from that dataset's subfolder names to the FINAL genre label
       you want in your model. Rename/merge as you like -- e.g. map both
       "pop" (GTZAN) and "bollypop" (IMG) to "pop" if you want them combined,
       or keep them as separate labels like "pop" and "bollypop".
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

from genre_features import extract_features

# ---------------------------------------------------------------------------
# Configure your datasets here. Each entry is a separate download, left in
# its own folder. "genre_map" keys are the subfolder names AS THEY APPEAR
# inside that dataset; values are the final label you want in your model.
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "path": "./Data/gtzan/genres_original",
        "genre_map": {
            "blues": "blues", "classical": "classical", "country": "country",
            "disco": "disco", "hiphop": "hiphop", "jazz": "jazz",
            "metal": "metal", "pop": "pop", "reggae": "reggae", "rock": "rock",
        },
    },
    {
        "path": "./Data/img",  # <-- adjust to actual folder name after unzip
        "genre_map": {
            "Bollypop": "bollypop", "Sufi": "sufi", "Ghazal": "ghazal",
            "Carnatic": "carnatic", "Hindustani": "hindustani_classical",
            "Semiclassical": "semiclassical",
        },
    },
    # Add more dataset entries here as you find/build them.
]

AUDIO_EXTENSIONS = (".wav", ".mp3", ".au")


def build_dataset():
    X, y = [], []
    for dataset in DATASETS:
        root = dataset["path"]
        genre_map = dataset["genre_map"]

        if not os.path.isdir(root):
            print(f"Skipping missing dataset root: {root}")
            continue

        for subfolder, final_label in genre_map.items():
            genre_dir = os.path.join(root, subfolder)
            if not os.path.isdir(genre_dir):
                print(f"  Skipping missing folder: {genre_dir}")
                continue

            for fname in os.listdir(genre_dir):
                if fname.lower().endswith(AUDIO_EXTENSIONS):
                    path = os.path.join(genre_dir, fname)
                    try:
                        X.append(extract_features(path))
                        y.append(final_label)
                        print(f"Processed [{final_label}] {path}")
                    except Exception as e:
                        print(f"Skipping {path}: {e}")

    return np.array(X), np.array(y)


def main():
    print("Extracting features from all configured datasets (this can take a while)...")
    X, y = build_dataset()

    if len(X) == 0:
        raise RuntimeError(
            "No audio files found. Check the 'path' and 'genre_map' entries "
            "in DATASETS -- they must match your actual downloaded folder names."
        )

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count}")

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
    print(f"Final genre set ({len(unique)} genres): {sorted(unique)}")


if __name__ == "__main__":
    main()