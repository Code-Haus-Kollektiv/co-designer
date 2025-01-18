import os
import json
import pandas as pd
import pickle
import xgboost as xgb

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""
    class Style:
        RESET_ALL = ""

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit=""):
        return iterable

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"
OUTPUT_FOLDER = r"./WP_2_Vector_Database/output"
MODEL_NAME = "testFile"
DEBUG_MODE = True

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------
def create_dated_output_subfolder(base_folder, model_name):
    """
    Creates a date-stamped subfolder inside 'base_folder' with the format:
    'YYYYMMdd_#_modelName'. If the subfolder exists, the # is incremented.
    Returns the path to the newly created subfolder.
    """
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    index = 1
    while True:
        subfolder_name = f"{date_str}_{index}_{model_name}"
        subfolder_path = os.path.join(base_folder, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            return subfolder_path
        index += 1

def load_json_files(folder_path):
    """Load and parse JSON files from a folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if DEBUG_MODE:
        # Limit to first 200 JSON files if in debug mode
        files = files[:200]
    components = []
    for filename in tqdm(files, desc=f"{Fore.CYAN}Reading JSON files{Style.RESET_ALL}", unit="file"):
        with open(os.path.join(folder_path, filename), encoding="utf-8") as file:
            data = json.load(file)
            components.extend(data.get("Components", []))
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Total components loaded: {len(components)}")
    return components

def find_downstream_components(current, all_comps):
    """Identify downstream components based on output parameters."""
    downstream_ids = set()
    for param in current.get("Parameters", []):
        if param.get("ParameterType") == "Output":
            connected_ids = param.get("ConnectedIds") or []
            for cid in connected_ids:
                for comp in all_comps:
                    if any(p.get("Id") == cid and p.get("ParameterType") == "Input"
                           for p in comp.get("Parameters", [])):
                        downstream_ids.add(comp.get("Id"))
    return list(downstream_ids)

def derive_next_component_name_and_id(current, all_comps):
    """Get the name and ID of the first downstream component."""
    downstream_ids = find_downstream_components(current, all_comps)
    for comp in all_comps:
        if comp.get("Id") in downstream_ids:
            return f"{comp.get('Id')}|{comp.get('Name')}"
    return None

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
        "NextComponent": comp.get("NextComponent", None)
    }

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} JSON_FOLDER path: {JSON_FOLDER}")
    components = load_json_files(JSON_FOLDER)
    
    # First, derive the next component for each item
    for comp in tqdm(components, desc=f"{Fore.CYAN}Deriving next components{Style.RESET_ALL}", unit="comp"):
        comp["NextComponent"] = derive_next_component_name_and_id(comp, components)
    
    # Filter out any components that have NO output parameters
    filtered_components = []
    for c in components:
        # Check if component has at least one parameter of type "Output"
        has_output_param = any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))
        if has_output_param:
            filtered_components.append(c)
    
    # Convert to DataFrame
    features = [extract_features(comp) for comp in filtered_components]
    df = pd.DataFrame(features)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} DataFrame shape (after filtering): {df.shape}")

    # Optionally remove rows where NextComponent is None
    df = df[df["NextComponent"].notna()].copy()
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} DataFrame shape (no None NextComponent): {df.shape}\n")

    # Create output subfolder with date stamping
    output_subfolder = create_dated_output_subfolder(OUTPUT_FOLDER, MODEL_NAME)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saving outputs to: {output_subfolder}")

    # Encode categorical variables
    encoders = {col: LabelEncoder() for col in ["Name", "InputParamIds", "NextComponent"]}

    # Fill NaN for Name, InputParamIds just in case
    df["Name"] = df["Name"].fillna("Unknown")
    df["InputParamIds"] = df["InputParamIds"].fillna("Unknown")

    for col, encoder in encoders.items():
        df[col] = df[col].astype(str)
        df[f"{col}_encoded"] = encoder.fit_transform(df[col])
        
        # Save encoder in the subfolder
        encoder_path = os.path.join(output_subfolder, f"{col}_encoder.pkl")
        with open(encoder_path, "wb") as f:
            pickle.dump(encoder, f)
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved {col} encoder to {encoder_path}")
    
    feature_cols = ["Name_encoded", "NumParams", "NumInput", "NumOutput", "InputParamIds_encoded"]
    X = df[feature_cols]
    y = df["NextComponent_encoded"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost parameters: fully utilize CPU cores
    params = {
        "objective": "multi:softmax",
        "num_class": len(encoders["NextComponent"].classes_),
        "tree_method": "hist",    # Fast, histogram-based algorithm (CPU)
        "nthread": -1     ,        # Use all available CPU cores
        # If you have a compatible GPU, you can use:
        # "tree_method": "gpu_hist",
        # "predictor": "gpu_predictor"
        # and optionally  "gpu_id": 0 (for the first GPU).
    }

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Training XGBoost model with full CPU utilization...")
    model = xgb.train(params, dtrain, num_boost_round=20)

    # Save the model in JSON format inside the subfolder
    model_path = os.path.join(output_subfolder, f"xgboost_model_{MODEL_NAME}.json")
    model.save_model(model_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Model saved as {model_path}")

    # Evaluate Model
    predictions = model.predict(dtest)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Test Accuracy: {accuracy:.4f}")

    # Example Prediction (if data is available)
    if not X_test.empty:
        sample_index = 0
        test_sample_features = X_test.iloc[sample_index]
        predicted_label_encoded = int(predictions[sample_index])
        actual_label_encoded = int(y_test.iloc[sample_index])
        
        predicted_label = encoders['NextComponent'].inverse_transform([predicted_label_encoded])[0]
        actual_label = encoders['NextComponent'].inverse_transform([actual_label_encoded])[0]
        
        print(f"{Fore.YELLOW}[SAMPLE FEATURES]{Style.RESET_ALL} {test_sample_features.to_dict()}")
        print(f"{Fore.YELLOW}[PREDICTION]{Style.RESET_ALL} Predicted: {predicted_label}, Actual: {actual_label}")

if __name__ == "__main__":
    main()
