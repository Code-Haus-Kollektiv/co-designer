import xgboost as xgb
import pandas as pd
import json
import os
import pickle
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Conditional imports for optional libraries
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""
    class Style:
        RESET_ALL = ""

# Constants
OUTPUT_FOLDER = r"./WP_2_Vector_Database/output"
MODEL_NAME = "testFile"
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"

# Function to load the model
def load_model(model_path):
    """Load a saved XGBoost model."""
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

# Function to extract features from a component
def extract_features(comp):
    """Extract features for a single component."""
    params = comp.get("Parameters", [])
    input_params = [p for p in params if p.get("ParameterType") == "Input"]
    return {
        "Id": comp.get("Id"),
        "Name": comp.get("Name", ""),
        "NumParams": len(params),
        "NumInput": len(input_params),
        "NumOutput": len(params) - len(input_params),
        "InputParamIds": ",".join([p.get("Id", "") for p in input_params]),
        "NextComponent": "Unknown"  # Default value if not present
    }

# Function to load JSON files and extract features
def load_json_files(folder_path, num_files=10):
    """Load and parse multiple JSON files to extract features."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")][:num_files]
    components = []
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
            components.extend(data.get("Components", []))
    features = [extract_features(comp) for comp in components]
    return pd.DataFrame(features)

# Function to load saved encoders
def load_encoders(output_folder, columns):
    """Load encoders for specified columns."""
    encoders = {}
    for col in columns:
        encoder_path = os.path.join(output_folder, f"{col}_encoder.pkl")
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                encoders[col] = pickle.load(f)
                print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded {col} encoder from {encoder_path}")
        else:
            raise FileNotFoundError(f"{col} encoder not found at {encoder_path}")
    return encoders

# Function to test the model
def test_model(model, X_test, y_test, encoders):
    """Test the loaded model with provided data."""
    dtest = xgb.DMatrix(X_test)
    predictions = model.predict(dtest)

    # Handle unseen predictions
    seen_labels = set(range(len(encoders['NextComponent'].classes_)))
    unseen_labels = set(predictions.astype(int)) - seen_labels
    if unseen_labels:
        print(f"{Fore.YELLOW}[WARNING]{Style.RESET_ALL} Unseen labels in predictions: {unseen_labels}")

    # Map predictions back to labels
    predicted_labels = [
        encoders["NextComponent"].inverse_transform([pred])[0] if pred in seen_labels else "Unknown"
        for pred in predictions.astype(int)
    ]
    actual_labels = encoders["NextComponent"].inverse_transform(
        [label for label in y_test if label in seen_labels]
    )

    # Evaluate model
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted", zero_division=1)
    try:
        report = classification_report(
            y_test, predictions, target_names=encoders["NextComponent"].classes_, 
            labels=range(len(encoders["NextComponent"].classes_)), zero_division=1
        )
    except ValueError as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {e}")
        report = "Classification report could not be generated due to mismatch in class labels."

    print(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} Test Accuracy: {accuracy:.4f}")
    print(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} Test F1 Score: {f1:.4f}")
    print(f"{Fore.CYAN}\nClassification Report:\n{Style.RESET_ALL}{report}")

    # Display example predictions
    for i in range(min(5, len(X_test))):
        print(f"{Fore.MAGENTA}Sample {i + 1}:{Style.RESET_ALL}")
        print(f"  Features: {X_test.iloc[i].to_dict()}")
        print(f"  Predicted: {predicted_labels[i]}, Actual: {actual_labels[i]}\n")

# Load model
model_path = os.path.join(OUTPUT_FOLDER, f"xgboost_model_{MODEL_NAME}.json")
model = load_model(model_path)

# Load test data
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loading test data from {JSON_FOLDER}")
df_test = load_json_files(JSON_FOLDER, num_files=10)

# Load encoders
encoders = load_encoders(OUTPUT_FOLDER, ["Name", "InputParamIds", "NextComponent"])

# Apply encoders to test data
for col, encoder in encoders.items():
    df_test[col] = df_test[col].fillna("Unknown")
    df_test[f"{col}_encoded"] = encoder.transform(df_test[col].astype(str))

# Prepare test data
feature_cols = ["Name_encoded", "NumParams", "NumInput", "NumOutput", "InputParamIds_encoded"]
X_test = df_test[feature_cols]
y_test = df_test["NextComponent_encoded"]

# Test the model
test_model(model, X_test, y_test, encoders)
