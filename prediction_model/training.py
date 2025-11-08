"""
Training a model to predict emotional states from EEG data.
This module handles data loading, preprocessing, model training, and evaluation.
It is designed to work with EEG data collected from the Muse2 headband.
Model inputs: preprocessed EEG features.
Model outputs: predicted emotional states (e.g., happy, sad, neutral).

Model trained on labeled EEG datasets with emotional state annotations.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from signal_processing.EEGProcessor import EEGProcessor


DEFAULT_BANDS = (
    ("delta", (0.5, 4)),
    ("theta", (4, 8)),
    ("alpha", (8, 12)),
    ("beta", (12, 30)),
)


def prepare_csv_dataset(
    csv_path: Union[str, Path],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
    dropna: bool = True,
    output_dir: Union[str, Path, None] = None,
) -> dict:
    """
    Prepare a CSV dataset for training an EEG emotion classifier.

    Parameters
    ----------
    csv_path : Union[str, Path]
        Path to a CSV file containing EEG features and labels.
    test_size : float, optional
        Fraction of samples to use for testing (default is 0.2).
    random_state : int, optional
        Random seed for reproducibility (default is 42).
    stratify : bool, optional
        Whether to stratify the split by label (default is True).
    dropna : bool, optional
        Whether to drop rows containing missing values (default is True).
    output_dir : Union[str, Path, None], optional
        Directory to write output files (default is the parent directory of csv_path).

    Returns
    -------
    dict
        Dictionary containing information about the prepared dataset, including
        the shapes of the training and test sets, the number of features, the
        number of dropped rows, and the paths to the train, test, and label
        mapping CSV files.
    """

    dataset_path = Path(csv_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df.rename(columns=lambda c: c.strip().lstrip("# "), inplace=True)

    if "label" not in df.columns:
        raise ValueError("Dataset does not contain a 'label' column.")

    original_len = len(df)
    if dropna:
        df = df.dropna()

    feature_columns = [col for col in df.columns if col != "label"]
    if not feature_columns:
        raise ValueError("Dataset does not contain any feature columns.")

    X = df[feature_columns].to_numpy(dtype=float)
    y_raw = df["label"].astype(str).to_numpy()

    label_encoder = LabelEncoder().fit(y_raw)
    y = label_encoder.transform(y_raw)

    stratify_target = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    destination = Path(output_dir) if output_dir else dataset_path.parent / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    prefix = dataset_path.stem

    npz_path = destination / f"{prefix}_split.npz"
    np.savez(
        npz_path,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_classes=label_encoder.classes_,
        feature_names=np.array(feature_columns),
    )

    def _write_split(split_X, split_y, suffix: str) -> Path:
        frame = pd.DataFrame(split_X, columns=feature_columns)
        frame["label"] = label_encoder.inverse_transform(split_y)
        split_path = destination / f"{prefix}_{suffix}.csv"
        frame.to_csv(split_path, index=False)
        return split_path

    train_csv_path = _write_split(X_train, y_train, "train_split")
    test_csv_path = _write_split(X_test, y_test, "test_split")

    mapping_path = destination / f"{prefix}_labels.json"
    with mapping_path.open("w", encoding="utf-8") as fh:
        json.dump({"classes": label_encoder.classes_.tolist()}, fh, indent=2)

    return {
        "train_shape": X_train.shape,
        "test_shape": X_test.shape,
        "feature_count": len(feature_columns),
        "dropped_rows": original_len - len(df),
        "train_csv": train_csv_path,
        "test_csv": test_csv_path,
        "npz": npz_path,
        "label_mapping": mapping_path,
    }


class EEGEmotionTrainer:
    """Lightweight EEG emotion classifier trainer.

    The class wraps data loading, feature extraction, model training, and
    evaluation so the calling code can stay lean. By default it relies on a
    logistic regression classifier with feature scaling, which keeps the model
    small and provides fast predictions even on low-power hardware.
    """

    def __init__(
        self,
        *,
        bands: Sequence[Tuple[str, Tuple[float, float]]] = DEFAULT_BANDS,
        test_size: float = 0.2,
        random_state: int = 42,
        model: Pipeline | None = None,
    ) -> None:
        self.band_sequence = tuple(bands)
        self.band_dict = OrderedDict(self.band_sequence)
        self.band_names = tuple(self.band_dict.keys())
        self.test_size = test_size
        self.random_state = random_state
        self.processor = EEGProcessor()
        self.model = model or Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=400,
                    ),
                ),
            ]
        )
        self.history = {}

    def load_feature_dataset(
        self, dataset_path: Union[str, Path], *, delimiter: str = ","
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load precomputed feature dataset from numpy/csv files.

        Parameters
        ----------
        dataset_path:
            Path to a ``.npz``, ``.npy`` or ``.csv`` file containing features.
        delimiter:
            Column delimiter to use when ingesting CSV-style files.

        Returns
        -------
        tuple of (X, y)
        """

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        suffix = dataset_path.suffix.lower()

        if suffix == ".npz":
            loaded = np.load(dataset_path)
            X, y = loaded["X"], loaded["y"]
        elif suffix == ".npy":
            loaded = np.load(dataset_path, allow_pickle=True)
            if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape[0] == 2:
                X, y = loaded.tolist()
            else:
                raise ValueError(".npy datasets must store [X, y] in the first dimension.")
        elif suffix in {".csv", ".txt"}:
            raw = np.loadtxt(dataset_path, delimiter=delimiter, dtype=float)
            X, y = raw[:, :-1], raw[:, -1]
        else:
            raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")

        return np.asarray(X, dtype=float), np.asarray(y)

    def extract_features(self, eeg_segment: np.ndarray, *, fs: int = 256) -> np.ndarray:
        """
        Extract features from a raw EEG segment.

        Parameters
        ----------
        eeg_segment : array_like
            Raw EEG segment, shape: (samples, channels)
        fs : int, optional
            Sampling frequency in Hz (default is 256)

        Returns
        -------
        features : array_like
            Extracted features, shape: (n_features,)
        """
        segment = np.asarray(eeg_segment, dtype=float)
        if segment.ndim != 2:
            raise ValueError("Expected EEG segment with shape (samples, channels).")

        filtered = self.processor.apply_filtering(segment, fs=fs)
        psd, freqs = self.processor.convert_to_psd(filtered, fs=fs)
        band_powers = self.processor.extract_bands(psd, freqs, bands=self.band_dict)

        features = []
        for name in self.band_names:
            absolute = band_powers[name]
            relative = absolute / (np.sum(absolute) + 1e-8)
            features.append(np.concatenate([absolute, relative], axis=0))

        return np.log1p(np.concatenate(features, axis=0))

    def build_feature_matrix(
        self, segments: Iterable[np.ndarray], *, fs: int = 256
    ) -> np.ndarray:
        """Extract feature matrix from multiple EEG segments."""

        feature_list = [self.extract_features(segment, fs=fs) for segment in segments]
        return np.vstack(feature_list)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        validation: bool = True,
        stratify: bool = True,
    ) -> dict:
        """Fit the model on the provided feature matrix.

        Returns a dictionary containing accuracy metrics. When ``validation`` is
        true, the data is internally split and accuracy is reported for both
        train and validation sets.
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if validation and len(np.unique(y)) > 1:
            stratify_target = y if stratify else None
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify_target,
            )
            self.model.fit(X_train, y_train)
            train_acc = float(accuracy_score(y_train, self.model.predict(X_train)))
            val_preds = self.model.predict(X_val)
            val_acc = float(accuracy_score(y_val, val_preds))
            report = classification_report(y_val, val_preds, zero_division=0)
            self.history["validation_report"] = report
            metrics = {"train_accuracy": train_acc, "val_accuracy": val_acc}
        else:
            self.model.fit(X, y)
            train_acc = float(accuracy_score(y, self.model.predict(X)))
            metrics = {"train_accuracy": train_acc}

        self.history["last_metrics"] = metrics
        return metrics

    def train_from_segments(
        self,
        segments: Sequence[np.ndarray],
        labels: Sequence[int],
        *,
        fs: int = 256,
        **kwargs,
    ) -> dict:
        """Ffeaturize raw EEG segments and train."""

        X = self.build_feature_matrix(segments, fs=fs)
        y = np.asarray(labels)
        return self.train(X, y, **kwargs)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate the fitted model on hold-out data."""

        if not hasattr(self.model, "predict"):
            raise RuntimeError("Model is not trained. Call train() first.")

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        predictions = self.model.predict(X)
        accuracy = float(accuracy_score(y, predictions))
        report = classification_report(y, predictions, zero_division=0)
        return {"accuracy": accuracy, "report": report}

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict labels for precomputed feature vectors."""

        if not hasattr(self.model, "predict"):
            raise RuntimeError("Model is not trained. Call train() first.")

        features = np.asarray(features, dtype=float)
        return self.model.predict(features)

    def predict_emotion(self, eeg_segment: np.ndarray, *, fs: int = 256) -> np.ndarray:
        """Predict emotion directly from a raw EEG segment."""

        features = self.extract_features(eeg_segment, fs=fs)
        return self.predict(features.reshape(1, -1))

    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the trained model to disk.
        
        Parameters
        ----------
        model_path : Union[str, Path]
            Path where the model should be saved (e.g., 'model.pkl')
        """
        if not hasattr(self.model, "predict"):
            raise RuntimeError("Model is not trained. Call train() first.")
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained model from disk.
        
        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the saved model file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = joblib.load(model_path)
        print(f"Model loaded from: {model_path}")


def run_training_pipeline(**kwargs):
    trainer = EEGEmotionTrainer()
    trainer.train_from_segments(**kwargs)


if __name__ == "__main__":
    print("=" * 60)
    print("EEG Emotion Classification Training")
    print("=" * 60)
    
    data_dir = Path(__file__).parent / "train_data" / "processed"
    train_csv = data_dir / "train_train_split.csv"
    test_csv = data_dir / "train_test_split.csv"
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test dataset not found: {test_csv}")
    
    print(f"\nLoading training data from: {train_csv}")
    print(f"Loading test data from: {test_csv}")
    
    # Load datasets
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    
    print(f"\nTraining set shape: {train_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    
    # Separate features and labels
    X_train = train_df.drop('label', axis=1).values
    y_train_raw = train_df['label'].values
    
    X_test = test_df.drop('label', axis=1).values
    y_test_raw = test_df['label'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    print(f"\nLabel classes: {label_encoder.classes_}")
    print(f"Class distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, count in zip(label_encoder.classes_, counts):
        print(f"  {cls}: {count}")
    
    # Initialize trainer
    print("\n" + "=" * 60)
    print("Training Model")
    print("=" * 60)
    trainer = EEGEmotionTrainer(random_state=42)
    
    # Train model 
    print("\nTraining on preprocessed features...")
    train_metrics = trainer.train(X_train, y_train, validation=False)
    print(f"Training accuracy: {train_metrics['train_accuracy']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating Model")
    print("=" * 60)
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(test_metrics['report'])
    
    # Save the trained model and label encoder
    print("\n" + "=" * 60)
    print("Saving Model")
    print("=" * 60)
    
    model_dir = Path(__file__).parent / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "eeg_emotion_classifier.pkl"
    trainer_path = model_dir / "eeg_emotion_trainer.pkl"
    label_encoder_path = model_dir / "label_encoder.pkl"
    
    # Save just the pipeline (for backward compatibility)
    trainer.save_model(model_path)
    
    # Save the entire trainer object (includes processor and feature extraction)
    joblib.dump(trainer, trainer_path)
    print(f"Full trainer saved to: {trainer_path}")
    
    joblib.dump(label_encoder, label_encoder_path)
    print(f"Label encoder saved to: {label_encoder_path}")
    
    # Save model metadata
    metadata = {
        "train_accuracy": train_metrics['train_accuracy'],
        "test_accuracy": test_metrics['accuracy'],
        "label_classes": label_encoder.classes_.tolist(),
        "n_features": X_train.shape[1],
        "train_samples": X_train.shape[0],
        "test_samples": X_test.shape[0],
    }
    metadata_path = model_dir / "model_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Model metadata saved to: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
