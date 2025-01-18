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
DEBUG_FILE_LIMIT = 100 if DEBUG_MODE else None

# ----------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------
def create_dated_output_subfolder(base_folder, model_name):
    """Create a date-stamped subfolder inside 'base_folder'."""
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
    if DEBUG_FILE_LIMIT:
        files = files[:DEBUG_FILE_LIMIT]
    components = []
    for filename in tqdm(files, desc=f"{Fore.CYAN}Reading JSON files{Style.RESET_ALL}", unit="file"):
        with open(os.path.join(folder_path, filename), encoding="utf-8") as file:
            data = json.load(file)
            components.extend(data.get("Components", []))
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Total components loaded: {len(components)}")
    return components

# ----------------------------------------------------------------------------
# Build Input/Output Maps
# ----------------------------------------------------------------------------
def build_input_map(components):
    """
    input_map[param_id] = list of comp_ids that have 'param_id' as an INPUT param
    """
    from collections import defaultdict
    in_map = defaultdict(list)
    for comp in components:
        cid = comp.get("Id")
        for p in comp.get("Parameters", []):
            if p.get("ParameterType") == "Input":
                pid = p.get("Id")
                if pid and cid:
                    in_map[pid].append(cid)
    return dict(in_map)

def build_output_map(components):
    """
    output_map[param_id] = list of comp_ids that produce 'param_id' as an OUTPUT
    """
    from collections import defaultdict
    out_map = defaultdict(list)
    for comp in components:
        cid = comp.get("Id")
        for p in comp.get("Parameters", []):
            if p.get("ParameterType") == "Output":
                pid = p.get("Id")
                if pid and cid:
                    out_map[pid].append(cid)
    return dict(out_map)

def build_comp_lookup(components):
    """Lookup by comp_id -> entire component object."""
    return {c["Id"]: c for c in components if c.get("Id")}

# ----------------------------------------------------------------------------
# Multi-upstream / Single-downstream
# ----------------------------------------------------------------------------
def get_upstream_ids(current_comp, output_map):
    """Return ALL upstream IDs (could be multiple) for 'current_comp'."""
    cid = current_comp.get("Id")
    if not cid:
        return []
    # For each INPUT param in 'current_comp', gather all producers
    input_param_ids = [
        p["Id"]
        for p in current_comp.get("Parameters", [])
        if p.get("ParameterType") == "Input" and p.get("Id")
    ]
    ups = set()
    for pid in input_param_ids:
        if pid in output_map:
            for producer_id in output_map[pid]:
                ups.add(producer_id)
    return list(ups)

def get_downstream_ids(current_comp, input_map):
    """Return ALL downstream IDs for 'current_comp'."""
    cid = current_comp.get("Id")
    if not cid:
        return []
    # For each OUTPUT param in 'current_comp', gather all consumers
    output_param_ids = [
        p["Id"]
        for p in current_comp.get("Parameters", [])
        if p.get("ParameterType") == "Output" and p.get("Id")
    ]
    downs = set()
    for pid in output_param_ids:
        if pid in input_map:
            for consumer_id in input_map[pid]:
                downs.add(consumer_id)
    return list(downs)

def extract_numeric_features(comp):
    """Extract numeric features like #input, #output, plus name."""
    params = comp.get("Parameters", [])
    input_params = [p for p in params if p.get("ParameterType") == "Input"]
    return {
        "Name": comp.get("Name", "Unknown"),
        "NumParams": len(params),
        "NumInput": len(input_params),
        "NumOutput": len(params) - len(input_params)
    }

def aggregate_upstream_features(upstream_ids, comp_lookup):
    """
    Given multiple upstream IDs, combine them into a single row's worth of data.
    For example:
      - Concatenate all GUIDs into "GUID1;GUID2;..."
      - Concatenate all names into "Name1;Name2;..."
      - Sum numeric features like #params, #input, etc.
    """
    if not upstream_ids:
        return {
            "AllUpstreamGUIDs": "",
            "AllUpstreamNames": "",
            "SumUpstreamNumParams": 0,
            "SumUpstreamNumInput": 0,
            "SumUpstreamNumOutput": 0
        }

    guids = []
    names = []
    sum_num_params = 0
    sum_num_input = 0
    sum_num_output = 0

    for uid in upstream_ids:
        up_comp = comp_lookup.get(uid, {})
        feats = extract_numeric_features(up_comp)
        guids.append(uid)
        names.append(feats["Name"])
        sum_num_params += feats["NumParams"]
        sum_num_input += feats["NumInput"]
        sum_num_output += feats["NumOutput"]

    return {
        "AllUpstreamGUIDs": ";".join(guids),
        "AllUpstreamNames": ";".join(names),
        "SumUpstreamNumParams": sum_num_params,
        "SumUpstreamNumInput": sum_num_input,
        "SumUpstreamNumOutput": sum_num_output
    }

def main():
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} JSON_FOLDER path: {JSON_FOLDER}")
    components = load_json_files(JSON_FOLDER)
    
    # Build maps/lookups
    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # Filter: keep only comps that produce at least one OUTPUT
    filtered = [
        c for c in components
        if any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))
    ]

    data_rows = []
    for current_comp in tqdm(filtered, desc=f"{Fore.CYAN}Building dataset{Style.RESET_ALL}", unit="comp"):
        current_id = current_comp.get("Id")
        if not current_id:
            continue

        # 1) Collect multiple upstream IDs
        upstream_ids = get_upstream_ids(current_comp, out_map)
        if not upstream_ids:
            continue

        # 2) Collect multiple downstream IDs
        downstream_ids = get_downstream_ids(current_comp, in_map)
        if not downstream_ids:
            continue

        # 3) Create a single aggregated upstream feature set
        upstream_agg = aggregate_upstream_features(upstream_ids, comp_lookup)

        # 4) We'll pick exactly ONE downstream to serve as label
        first_down_id = downstream_ids[0]
        down_comp = comp_lookup.get(first_down_id, {})
        down_label = f"{first_down_id}|{down_comp.get('Name','Unknown')}"

        # 5) Extract numeric features for the current component
        current_feats = extract_numeric_features(current_comp)

        # Combine everything into one row
        row = {
            # Upstream (aggregated)
            "UpstreamGUIDs": upstream_agg["AllUpstreamGUIDs"],
            "UpstreamNames": upstream_agg["AllUpstreamNames"],
            "UpstreamNumParamsSum": upstream_agg["SumUpstreamNumParams"],
            "UpstreamNumInputSum": upstream_agg["SumUpstreamNumInput"],
            "UpstreamNumOutputSum": upstream_agg["SumUpstreamNumOutput"],

            # Current
            "CurrentGUID": current_id,
            "CurrentName": current_feats["Name"],
            "CurrentNumParams": current_feats["NumParams"],
            "CurrentNumInput": current_feats["NumInput"],
            "CurrentNumOutput": current_feats["NumOutput"],

            # Single-label for downstream
            "DownstreamGUIDName": down_label
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Final dataset shape: {df.shape}")
    if df.empty:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid data found. Exiting.")
        return

    # Create output folder
    output_subfolder = create_dated_output_subfolder(OUTPUT_FOLDER, MODEL_NAME)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saving outputs to: {output_subfolder}")

    # ------------------------
    # Encoding
    # ------------------------
    cat_cols = ["UpstreamGUIDs", "UpstreamNames", "CurrentGUID", "CurrentName"]
    label_col = "DownstreamGUIDName"

    encoders = {}
    for col in cat_cols + [label_col]:
        enc = LabelEncoder()
        df[col] = df[col].astype(str)
        df[f"{col}_encoded"] = enc.fit_transform(df[col])
        encoders[col] = enc

        # Save the encoder
        enc_path = os.path.join(output_subfolder, f"{col}_encoder.pkl")
        with open(enc_path, "wb") as f:
            pickle.dump(enc, f)
            print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved encoder for {col} -> {enc_path}")

    # Prepare features
    feature_cols = [
        "UpstreamGUIDs_encoded",
        "UpstreamNames_encoded",
        "UpstreamNumParamsSum",
        "UpstreamNumInputSum",
        "UpstreamNumOutputSum",

        "CurrentGUID_encoded",
        "CurrentName_encoded",
        "CurrentNumParams",
        "CurrentNumInput",
        "CurrentNumOutput"
    ]
    X = df[feature_cols]
    y = df[f"{label_col}_encoded"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # ------------------------
    # XGBoost parameters
    # ------------------------
    # Use multi:softprob to get probabilities over classes (for top-5).
    params = {
        "objective": "multi:softprob",
        "num_class": len(encoders[label_col].classes_),
        "tree_method": "hist",
        "nthread": -1
    }

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Training XGBoost (single-downstream) model...")
    model = xgb.train(params, dtrain, num_boost_round=30)

    # Save model
    model_path = os.path.join(output_subfolder, f"xgboost_model_{MODEL_NAME}.json")
    model.save_model(model_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Model saved -> {model_path}")

    # ------------------------
    # Prediction & Evaluation
    # ------------------------
    # pred_proba will be N x num_class array of probabilities
    pred_proba = model.predict(dtest)

    # For top-1 accuracy, pick the class with the highest probability
    preds_top1 = pred_proba.argmax(axis=1)
    acc = accuracy_score(y_test, preds_top1)
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Top-1 Accuracy: {acc:.4f}")

    # Example inference: get top-5 predictions with scores (first row by position)
    if not X_test.empty:
        sample_idx = 0  # pick the first row in the test set

        # --- Decode the Current Component for this row ---
        row_feats = X_test.iloc[sample_idx]  # numeric-encoded features for this row

        currentGUID_enc = int(row_feats["CurrentGUID_encoded"])
        currentName_enc = int(row_feats["CurrentName_encoded"])

        decoded_currentGUID = encoders["CurrentGUID"].inverse_transform([currentGUID_enc])[0]
        decoded_currentName = encoders["CurrentName"].inverse_transform([currentName_enc])[0]

        print(f"{Fore.BLUE}[CURRENT COMPONENT]{Style.RESET_ALL} "
            f"GUID: {decoded_currentGUID}, Name: {decoded_currentName}")

        # --- Now get the top-5 predictions ---
        row_probs = pred_proba[sample_idx]  # probabilities for this row
        top_5_idx = row_probs.argsort()[::-1][:5]
        top_5_scores = row_probs[top_5_idx]
        decoded_labels = encoders[label_col].inverse_transform(top_5_idx)

        print(f"\n{Fore.MAGENTA}[INFO]{Style.RESET_ALL} Top-5 predictions for test row {sample_idx}:")
        for rank, (lbl, score) in enumerate(zip(decoded_labels, top_5_scores), start=1):
            print(f"  Rank {rank} -> label: {lbl}, score: {score:.4f}")

        # Show the actual label
        actual_enc = y_test.iloc[sample_idx]
        actual_str = encoders[label_col].inverse_transform([actual_enc])[0]
        print(f"\n{Fore.GREEN}[ACTUAL LABEL]{Style.RESET_ALL} {actual_str}")


if __name__ == "__main__":
    main()
