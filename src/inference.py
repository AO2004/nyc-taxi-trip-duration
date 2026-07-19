# load the models first
import json
import joblib
from pathlib import Path

from src.feature_engineering import build_features

# 1. Dynamically locate the 'model' directory relative to this script
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR/'models'

print("Loading artifacts from:", MODEL_DIR)

# 2. Load the metadata file
metadata_path = MODEL_DIR/'metadata.json'
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# Extract feature lists from your metadata structure
FEATURES = metadata['features']
CATEGORICAL_FEATURES = metadata['categorical_features']
NUMERICAL_FEATURES = metadata['numerical_features']
print('-----------------------------------------------------------------')
print(f'Features: \n\t{FEATURES}\n')
print(f'Categorical features: \n\t{CATEGORICAL_FEATURES}\n')
print(f'Numerical features: \n\t{NUMERICAL_FEATURES}\n')
print('-----------------------------------------------------------------')

# 3. Load the model objects using fixed strings (bypassing the missing JSON keys)
try:
    model = joblib.load(MODEL_DIR/'main_model.joblib')
    preprocessor = joblib.load(MODEL_DIR/'preprocessor.joblib')
    kmeans = joblib.load(MODEL_DIR/'kmeans_model.joblib')
    print("All models loaded successfully!")
except FileNotFoundError as e:
    print(f"\n[Error] Make sure the file names match exactly in your 'models' directory: {e}")
    raise


def inference(data):

    # feature engineering
    build_features(data, kmeans)
    data = data[FEATURES]

    # preprocessing
    processed_data = preprocessor.transform(data)

    return model.predict(processed_data)