"""
Predict emotions from pre-extracted EEG features.
This works with the current trained model which expects 2548 features.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def predict_from_features(model_path: str, features: np.ndarray, label_encoder_path: str = None) -> tuple:
    """
    Predict emotion from pre-extracted EEG features.

    Parameters
    ----------
    model_path : str
        Path to the saved model file.
    features : np.ndarray
        Pre-extracted features matching the training format.
    label_encoder_path : str, optional
        Path to the saved label encoder. If provided, returns the class name.

    Returns
    -------
    tuple
        (predicted_class_index, predicted_class_name) if label_encoder is provided,
        otherwise (predicted_class_index, None)
    """
    # Load the trained pipeline
    pipeline = joblib.load(model_path)
    
    # Load label encoder if provided
    label_encoder = None
    if label_encoder_path:
        label_encoder = joblib.load(label_encoder_path)
    
    # Ensure features are in the correct shape
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    # Make prediction
    prediction = pipeline.predict(features)
    predicted_class = int(prediction[0])
    
    # Get class name if label encoder is available
    class_name = None
    if label_encoder is not None:
        class_name = label_encoder.inverse_transform([predicted_class])[0]
    
    return predicted_class, class_name


if __name__ == "__main__":
    print("=" * 60)
    print("EEG Emotion Prediction from Pre-extracted Features")
    print("=" * 60)

    data_dir = Path(__file__).parent / "saved_models" 
    model_file = data_dir / "eeg_emotion_classifier.pkl"
    label_encoder_file = data_dir / "label_encoder.pkl"
    
    # Load test data to get real examples
    test_csv = Path(__file__).parent / "train_data" / "processed" / "train_test_split.csv"
    
    if not test_csv.exists():
        print(f"\nError: Test data not found at {test_csv}")
        print("Please ensure the training data has been processed.")
        exit(1)
    
    test_df = pd.read_csv(test_csv)
    print(f"\nLoaded test data: {test_df.shape}")
    print(f"Available labels: {test_df['label'].unique()}")
    
    # Test predictions on a few samples from each class
    for label in test_df['label'].unique():
        print(f"\n{'-' * 60}")
        print(f"Testing samples with actual label: {label}")
        print('-' * 60)
        
        # Get first 3 samples of this class
        samples = test_df[test_df['label'] == label].head(3)
        
        for idx, (_, row) in enumerate(samples.iterrows(), 1):
            # Extract features (all columns except 'label')
            features = row.drop('label').values
            
            # Predict
            pred_class, pred_name = predict_from_features(
                model_file, 
                features, 
                label_encoder_file
            )
            
            # Check if prediction matches actual label
            match = "Correct" if pred_name == label else "False"
            print(f"  Sample {idx}: Predicted={pred_name}, Actual={label} {match}")
    
    print("\n" + "=" * 60)
    print("Prediction Complete!")
    print("=" * 60)
