import os
import json
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""
    class Style:
        RESET_ALL = ""

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
OUTPUT_FOLDER = r"./WP_2_Vector_Database/output"
MODEL_NAME = "testFile"
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"
TOP_K = 3   # Number of top predictions to retrieve (e.g., top 3)

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------
def load_model(model_path):
    """Load a saved XGBoost model from JSON or .model file."""
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Model loaded from {model_path}")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

def extract_features(comp):
    """
    Extract numeric/categorical features from a single component.
    'NextComponent' is set to 'Unknown' if not present, so we can still
    apply encoders consistently.
    """
    params = comp.get("Parameters", [])
    input_params = [p for p in params if p.get("ParameterType") == "Input"]
    return {
        "Id": comp.get("Id"),
        "Name": comp.get("Name", ""),
        "NumParams": len(params),
        "NumInput": len(input_params),
        "NumOutput": len(params) - len(input_params),
        "InputParamIds": ",".join([p.get("Id", "") for p in input_params]),
        "NextComponent": "Unknown"
    }

def load_json_files(folder_path, num_files=10):
    """
    Load multiple JSON files to build a test dataset.
    'num_files' can be increased or set to None to load all.
    """
    file_list = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if num_files is not None:
        file_list = file_list[:num_files]
    
    comps = []
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
            comps.extend(data.get("Components", []))
    features = [extract_features(comp) for comp in comps]
    return pd.DataFrame(features)

def load_encoders(output_folder, columns):
    """Load LabelEncoders for specified columns from .pkl files."""
    encoders = {}
    for col in columns:
        enc_path = os.path.join(output_folder, f"{col}_encoder.pkl")
        if os.path.exists(enc_path):
            with open(enc_path, "rb") as f:
                encoders[col] = pickle.load(f)
                print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded {col} encoder from {enc_path}")
        else:
            raise FileNotFoundError(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {col} encoder not found at {enc_path}")
    return encoders

def test_model(model, X_test, y_test, encoders, top_k=3):
    """
    Evaluate the model on (X_test, y_test).
    Also print top-k predicted labels for each sample (if the model was trained
    with 'multi:softprob' objective).
    """
    dtest = xgb.DMatrix(X_test)
    raw_preds = model.predict(dtest)
    
    # Check shape of raw_preds:
    #   1D if model uses 'multi:softmax' => single best label per row
    #   2D if model uses 'multi:softprob' => probability distribution per row
    if len(raw_preds.shape) == 1:
        # 'softmax' scenario => raw_preds is the best label for each sample
        preds_top1 = raw_preds.astype(int)
        # Evaluate
        accuracy = accuracy_score(y_test, preds_top1)
        f1 = f1_score(y_test, preds_top1, average="weighted", zero_division=1)
        print(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} Accuracy: {accuracy:.4f}")
        print(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} F1 Score: {f1:.4f}")

        # Classification report
        try:
            report = classification_report(
                y_test,
                preds_top1,
                target_names=encoders["NextComponent"].classes_,
                labels=range(len(encoders["NextComponent"].classes_)),
                zero_division=1,
            )
        except ValueError as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {e}")
            report = "Classification report could not be generated."
        print(f"{Fore.CYAN}Classification Report:\n{Style.RESET_ALL}{report}")

        # Decode predictions
        seen_labels = set(range(len(encoders["NextComponent"].classes_)))
        predicted_labels = [
            encoders["NextComponent"].inverse_transform([p])[0] if p in seen_labels else "Unknown"
            for p in preds_top1
        ]
        actual_labels = encoders["NextComponent"].inverse_transform(y_test.astype(int))

        # Show sample predictions
        for i in range(min(5, len(X_test))):
            print(f"\n{Fore.MAGENTA}[SAMPLE {i+1}]{Style.RESET_ALL}")
            print(f"Features: {X_test.iloc[i].to_dict()}")
            print(f"Predicted: {predicted_labels[i]}, Actual: {actual_labels[i]}")

    else:
        # 'softprob' scenario => raw_preds is (num_samples, num_classes)
        # We pick top-1 for accuracy/f1
        preds_top1 = raw_preds.argmax(axis=1).astype(int)
        accuracy = accuracy_score(y_test, preds_top1)
        f1 = f1_score(y_test, preds_top1, average="weighted", zero_division=1)
        print(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} Accuracy: {accuracy:.4f}")
        print(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} F1 Score: {f1:.4f}")

        # Classification report
        try:
            report = classification_report(
                y_test,
                preds_top1,
                target_names=encoders["NextComponent"].classes_,
                labels=range(len(encoders["NextComponent"].classes_)),
                zero_division=1,
            )
        except ValueError as e:
            print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {e}")
            report = "Classification report could not be generated."
        print(f"{Fore.CYAN}Classification Report:\n{Style.RESET_ALL}{report}")

        # Decode top-1 predictions
        seen_labels = set(range(len(encoders["NextComponent"].classes_)))
        predicted_labels_top1 = [
            encoders["NextComponent"].inverse_transform([p])[0] if p in seen_labels else "Unknown"
            for p in preds_top1
        ]
        # Decode actual
        actual_labels = encoders["NextComponent"].inverse_transform(y_test.astype(int))

        # Show sample predictions (including top-k).
        for i in range(min(5, len(X_test))):
            print(f"\n{Fore.MAGENTA}[SAMPLE {i+1}]{Style.RESET_ALL}")
            row_probs = raw_preds[i]
            top_k_indices = row_probs.argsort()[::-1][:top_k]
            top_k_scores = row_probs[top_k_indices]
            decoded_top_k = encoders["NextComponent"].inverse_transform(top_k_indices)

            print(f"Features: {X_test.iloc[i].to_dict()}")
            print(f"Actual: {actual_labels[i]}")
            print(f"Top-1 Predicted: {predicted_labels_top1[i]}")
            print(f"{Fore.BLUE}Top-{top_k} predictions{Style.RESET_ALL}:")
            for rank, (lbl_idx, score) in enumerate(zip(decoded_top_k, top_k_scores), start=1):
                print(f"  Rank {rank} -> label: {lbl_idx}, prob: {score:.4f}")

# ----------------------------------------------------------------------------
# MAIN SCRIPT
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    model_path = os.path.join(OUTPUT_FOLDER, f"xgboost_model_{MODEL_NAME}.json")
    model = load_model(model_path)
    
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loading test data from {JSON_FOLDER}")
    df_test = load_json_files(JSON_FOLDER, num_files=10)  # load a small subset for testing

    # Load encoders
    encoders = load_encoders(OUTPUT_FOLDER, ["Name", "InputParamIds", "NextComponent"])

    # Apply encoders
    for col in ["Name", "InputParamIds"]:
        df_test[col] = df_test[col].fillna("Unknown")
        df_test[f"{col}_encoded"] = encoders[col].transform(df_test[col].astype(str))

    # 'NextComponent' might not be known in real usage. If you do have it, encode it:
    if "NextComponent" in df_test.columns:
        df_test["NextComponent"] = df_test["NextComponent"].fillna("Unknown")
        df_test["NextComponent_encoded"] = encoders["NextComponent"].transform(df_test["NextComponent"].astype(str))
    else:
        # If NextComponent is not available, just create an empty column
        df_test["NextComponent_encoded"] = -1  # placeholders

    # Prepare data
    feature_cols = ["Name_encoded", "NumParams", "NumInput", "NumOutput", "InputParamIds_encoded"]
    X_test = df_test[feature_cols]

    # If NextComponent was loaded, you can do an evaluation:
    if "NextComponent_encoded" in df_test.columns:
        y_test = df_test["NextComponent_encoded"]
    else:
        y_test = pd.Series([-1]*len(X_test), index=X_test.index)

    # Evaluate
    test_model(model, X_test, y_test, encoders, top_k=TOP_K)
