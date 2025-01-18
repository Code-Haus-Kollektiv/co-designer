import os
import json
import pandas as pd
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
DEBUG_FILE_LIMIT = 1000 if DEBUG_MODE else None

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

def get_upstream_ids(current_comp, output_map):
    cid = current_comp.get("Id")
    if not cid:
        return []
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
    cid = current_comp.get("Id")
    if not cid:
        return []
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

        # 1) Gather multiple upstream IDs
        upstream_ids = get_upstream_ids(current_comp, out_map)
        if not upstream_ids:
            continue

        # 2) Gather multiple downstream IDs
        downstream_ids = get_downstream_ids(current_comp, in_map)
        if not downstream_ids:
            continue

        # 3) Aggregated upstream features
        upstream_agg = aggregate_upstream_features(upstream_ids, comp_lookup)

        # 4) Single label => first downstream
        first_down_id = downstream_ids[0]
        down_comp = comp_lookup.get(first_down_id, {})
        down_label = f"{first_down_id}|{down_comp.get('Name','Unknown')}"

        # 5) Current component numeric feats
        current_feats = extract_numeric_features(current_comp)

        row = {
            "UpstreamGUIDs": upstream_agg["AllUpstreamGUIDs"],
            "UpstreamNames": upstream_agg["AllUpstreamNames"],
            "SumUpstreamNumParams": upstream_agg["SumUpstreamNumParams"],
            "SumUpstreamNumInput": upstream_agg["SumUpstreamNumInput"],
            "SumUpstreamNumOutput": upstream_agg["SumUpstreamNumOutput"],

            "CurrentGUID": current_id,
            "CurrentName": current_feats["Name"],
            "CurrentNumParams": current_feats["NumParams"],
            "CurrentNumInput": current_feats["NumInput"],
            "CurrentNumOutput": current_feats["NumOutput"],

            "DownstreamGUIDName": down_label
        }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Final dataset shape: {df.shape}")
    if df.empty:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid data found. Exiting.")
        return

    output_subfolder = create_dated_output_subfolder(OUTPUT_FOLDER, MODEL_NAME)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saving outputs to: {output_subfolder}")

    # Categorical columns + label column
    cat_cols = ["UpstreamGUIDs", "UpstreamNames", "CurrentGUID", "CurrentName"]
    label_col = "DownstreamGUIDName"

    encoders = {}
    # We'll store JSON label map files (instead of pkl).
    def export_encoder_to_json(encoder, output_path):
        """Dump encoder.classes_ to JSON so C# can interpret them easily."""
        classes_list = encoder.classes_.tolist()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(classes_list, f, ensure_ascii=False, indent=2)

    for col in cat_cols + [label_col]:
        enc = LabelEncoder()
        df[col] = df[col].astype(str)
        df[f"{col}_encoded"] = enc.fit_transform(df[col])
        encoders[col] = enc

        # Save the encoder classes as JSON (array of strings).
        json_path = os.path.join(output_subfolder, f"{col}_encoder.json")
        export_encoder_to_json(enc, json_path)
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Created JSON label map for {col} -> {json_path}")

    # Prepare feature matrix X and label y
    feature_cols = [
        "UpstreamGUIDs_encoded",
        "UpstreamNames_encoded",
        "SumUpstreamNumParams",
        "SumUpstreamNumInput",
        "SumUpstreamNumOutput",
        "CurrentGUID_encoded",
        "CurrentName_encoded",
        "CurrentNumParams",
        "CurrentNumInput",
        "CurrentNumOutput"
    ]
    X = df[feature_cols]
    y = df[f"{label_col}_encoded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": len(encoders[label_col].classes_),
        "tree_method": "hist",
        "nthread": -1
    }

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Training XGBoost model...")
    model = xgb.train(params, dtrain, num_boost_round=30)

    # Save model in XGBoost JSON format
    model_path = os.path.join(output_subfolder, f"xgboost_model_{MODEL_NAME}.json")
    model.save_model(model_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Model saved -> {model_path}")

    # Evaluate
    pred_proba = model.predict(dtest)
    preds_top1 = pred_proba.argmax(axis=1)
    acc = accuracy_score(y_test, preds_top1)
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Top-1 Accuracy: {acc:.4f}")

    if not X_test.empty:
        sample_idx = 0
        row_feats = X_test.iloc[sample_idx]
        # Decode the "current" comp name
        currentGUID_enc = int(row_feats["CurrentGUID_encoded"])
        currentName_enc = int(row_feats["CurrentName_encoded"])

        decoded_currentGUID = encoders["CurrentGUID"].inverse_transform([currentGUID_enc])[0]
        decoded_currentName = encoders["CurrentName"].inverse_transform([currentName_enc])[0]
        print(f"{Fore.BLUE}[CURRENT COMPONENT]{Style.RESET_ALL} "
              f"GUID: {decoded_currentGUID}, Name: {decoded_currentName}")

        # Top-5 predictions
        row_probs = pred_proba[sample_idx]
        top_5_idx = row_probs.argsort()[::-1][:5]
        top_5_scores = row_probs[top_5_idx]
        decoded_labels = encoders[label_col].inverse_transform(top_5_idx)

        print(f"\n{Fore.MAGENTA}[INFO]{Style.RESET_ALL} Top-5 predictions for row {sample_idx}:")
        for rank, (lbl, score) in enumerate(zip(decoded_labels, top_5_scores), start=1):
            print(f"  Rank {rank} -> label: {lbl}, score: {score:.4f}")

        actual_enc = y_test.iloc[sample_idx]
        actual_str = encoders[label_col].inverse_transform([actual_enc])[0]
        print(f"\n{Fore.GREEN}[ACTUAL LABEL]{Style.RESET_ALL} {actual_str}")

if __name__ == "__main__":
    main()
