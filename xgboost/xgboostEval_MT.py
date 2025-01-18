import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime
from collections import defaultdict

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
# Paths and Constants
# ----------------------------------------------------------------------------
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"   # folder with test JSON files
MODEL_FOLDER = r"./WP_2_Vector_Database/output/20250118_3_testFile"               # folder containing model + encoders
DEBUG_FILE_LIMIT = 100                                       # set to an integer if you want to limit JSON processing
TOP_K = 5                                                     # how many top predictions to show

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------
def load_label_encoder_from_json(json_path):
    """
    Rebuild a LabelEncoder from a JSON file that contains the .classes_ array.
    """
    from sklearn.preprocessing import LabelEncoder
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Encoder JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        classes_list = json.load(f)

    le = LabelEncoder()
    le.classes_ = np.array(classes_list, dtype=object)
    return le

def load_encoders(encoder_folder, columns):
    """
    Given the list of columns that were label-encoded in training,
    load the corresponding JSON-based encoders.
    """
    encoders = {}
    for col in columns:
        json_path = os.path.join(encoder_folder, f"{col}_encoder.json")
        encoders[col] = load_label_encoder_from_json(json_path)
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded encoder for '{col}' from {json_path}")
    return encoders

def load_xgboost_model(model_folder):
    """
    Load the saved XGBoost model (JSON format) from training.
    Expects 'xgboost_model.json' inside 'model_folder'.
    """
    model_path = os.path.join(model_folder, "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model JSON not found: {model_path}")

    model = xgb.Booster()
    model.load_model(model_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} XGBoost model loaded from {model_path}")
    return model

def load_json_files(folder_path):
    """
    Load and parse all (or a limited number of) JSON files from a folder,
    returning a list of all 'Components' found.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if DEBUG_FILE_LIMIT:
        files = files[:DEBUG_FILE_LIMIT]

    all_components = []
    for filename in files:
        path = os.path.join(folder_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            comps = data.get("Components", [])
            all_components.extend(comps)

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Loaded {len(all_components)} total components from JSON.")
    return all_components

def build_input_map(components):
    """
    input_map[param_id] = list of comp_ids that have 'param_id' as an INPUT param
    """
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
    """
    Dict: comp_id -> entire component object
    """
    return {c["Id"]: c for c in components if c.get("Id")}

def get_upstream_ids(current_comp, output_map):
    """
    For a given component, find all component-IDs that produce an OUTPUT param
    matching one of current_comp's INPUT params.
    """
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
    """
    For a given component, find all component-IDs that consume as INPUT
    one of current_comp's OUTPUT params.
    """
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
    """
    Same numeric features used in training:
    - Name
    - NumParams
    - NumInput
    - NumOutput
    """
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
    Summation of numeric features from all upstream components:
      - AllUpstreamGUIDs
      - AllUpstreamNames
      - SumUpstreamNumParams
      - SumUpstreamNumInput
      - SumUpstreamNumOutput
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

# ----------------------------------------------------------------------------
# Main Inference Function
# ----------------------------------------------------------------------------
def main():
    # 1) Load the XGBoost model directly from the folder
    model = load_xgboost_model(MODEL_FOLDER)

    # 2) Load the JSON-based label encoders
    #    Must match the columns used in training:
    cat_cols = ["UpstreamGUIDs", "UpstreamNames", "CurrentGUID", "CurrentName"]
    label_col = "DownstreamGUIDName"
    all_encoded_cols = cat_cols + [label_col]
    encoders = load_encoders(MODEL_FOLDER, all_encoded_cols)

    # 3) Load test data from JSON
    components = load_json_files(JSON_FOLDER)
    if not components:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No components found. Exiting.")
        return

    # 4) Build in_map, out_map, comp_lookup
    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # 5) Filter: only keep comps that produce at least one OUTPUT
    filtered = [
        c for c in components
        if any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))
    ]
    if not filtered:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} After filtering, no valid components remain.")
        return

    # 6) Rebuild the same rows as training
    data_rows = []
    for current_comp in filtered:
        current_id = current_comp.get("Id")
        if not current_id:
            continue

        # Upstream
        upstream_ids = get_upstream_ids(current_comp, out_map)
        if not upstream_ids:
            # If there's no upstream, skip or handle differently
            continue

        # Downstream
        downstream_ids = get_downstream_ids(current_comp, in_map)
        if downstream_ids:
            first_down_id = downstream_ids[0]
            down_comp = comp_lookup.get(first_down_id, {})
            down_label = f"{first_down_id}|{down_comp.get('Name','Unknown')}"
        else:
            down_label = "Unknown"

        # Upstream aggregated features
        upstream_agg = aggregate_upstream_features(upstream_ids, comp_lookup)

        # Current comp numeric feats
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

            # actual label (if known). If unknown, use "Unknown"
            "DownstreamGUIDName": down_label
        }
        data_rows.append(row)

    df_test = pd.DataFrame(data_rows)
    if df_test.empty:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid rows to predict. Exiting.")
        return

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Test dataset shape: {df_test.shape}")

    # 7) Apply the same label encoding as training
    #    Feature columns used in training:
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
    encoded_label_col = "DownstreamGUIDName_encoded"

    for col in ["UpstreamGUIDs", "UpstreamNames", "CurrentGUID", "CurrentName", label_col]:
        df_test[col] = df_test[col].fillna("Unknown").astype(str)
        enc = encoders[col]
        df_test[f"{col}_encoded"] = enc.transform(df_test[col])

    # 8) Build X_test
    X_test = df_test[feature_cols]

    # If you do have ground-truth, build y_test. Otherwise, we just store placeholders
    y_test = df_test[encoded_label_col] if encoded_label_col in df_test else pd.Series([-1]*len(df_test))

    # 9) Predict
    dtest = xgb.DMatrix(X_test)
    pred_proba = model.predict(dtest)  # shape = (N, #classes) if "multi:softprob"

    # 10) Show top-K predictions
    if pred_proba.ndim == 1:
        # If the model was trained with multi:softmax or a single-class objective,
        # pred_proba is actually the predicted label indices
        preds_top1 = pred_proba.astype(int)
    else:
        # "softprob" => probability distribution
        preds_top1 = pred_proba.argmax(axis=1)

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Sample predictions:")
    for i in range(min(5, len(df_test))):
        row = df_test.iloc[i]
        row_probs = pred_proba[i] if pred_proba.ndim > 1 else None

        print(f"\n{Fore.MAGENTA}--- SAMPLE {i} ---{Style.RESET_ALL}")
        print(f"CurrentGUID: {row['CurrentGUID']}, CurrentName: {row['CurrentName']}")

        if row_probs is not None:
            # top-K from probability distribution
            top_k_idx = row_probs.argsort()[::-1][:TOP_K]
            top_k_scores = row_probs[top_k_idx]
            label_enc = encoders["DownstreamGUIDName"]
            decoded_top_k = label_enc.inverse_transform(top_k_idx)

            for rank, (lbl, score) in enumerate(zip(decoded_top_k, top_k_scores), start=1):
                print(f"   Rank {rank} -> {lbl}, score={score:.4f}")

            # actual (if known and not "Unknown")
            actual_enc = y_test.iloc[i]
            if actual_enc >= 0:
                actual_str = label_enc.inverse_transform([actual_enc])[0]
                print(f"   Actual label: {actual_str}")
        else:
            # single label approach
            best_lbl_idx = preds_top1[i]
            label_enc = encoders["DownstreamGUIDName"]
            best_lbl_str = label_enc.inverse_transform([best_lbl_idx])[0]
            print(f"   Predicted label: {best_lbl_str}")

            # actual (if known)
            actual_enc = y_test.iloc[i]
            if actual_enc >= 0:
                actual_str = label_enc.inverse_transform([actual_enc])[0]
                print(f"   Actual label: {actual_str}")

if __name__ == "__main__":
    main()
