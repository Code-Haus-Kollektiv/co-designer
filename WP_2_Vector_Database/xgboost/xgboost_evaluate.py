import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import argparse

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)
except ImportError:
    class Fore:
        RED = GREEN = YELLOW = CYAN = MAGENTA = ""
    class Style:
        RESET_ALL = ""

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report

from xgboost_utilities import (
    load_json_files,
    build_input_map,
    build_output_map,
    build_comp_lookup,
    get_upstream_ids,
    get_downstream_ids,
    extract_numeric_features
)

############################################
# If you want to limit how many JSON files
# are loaded during evaluation, set a value
############################################
EVAL_FILE_LIMIT = 50

def load_label_encoder_from_json(json_path):
    """Utility to load LabelEncoder classes_ from a JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Encoder JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        classes_list = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes_list, dtype=object)
    return le

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", required=True, help="Path to folder with xgboost_model.json + encoders")
    parser.add_argument("--test_json_folder", required=True, help="Path to folder with JSON test data")
    args = parser.parse_args()

    MODEL_FOLDER = args.model_folder
    TEST_JSON_FOLDER = args.test_json_folder

    # 1) Load XGBoost model
    model_path = os.path.join(MODEL_FOLDER, "xgboost_model.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"XGBoost model not found at {model_path}")

    model = xgb.Booster()
    model.load_model(model_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded XGBoost model from {model_path}")

    # 2) Load label encoders for CurrentGUID, CurrentName, DownstreamGUIDName
    cat_cols = ["CurrentGUID", "CurrentName", "DownstreamGUIDName"]
    encoders = {}
    for col in cat_cols:
        enc_path = os.path.join(MODEL_FOLDER, f"{col}_encoder.json")
        enc = load_label_encoder_from_json(enc_path)
        encoders[col] = enc
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded encoder for '{col}' from {enc_path}")

    # 3) Load top GUIDs/Names config (for multi-hot upstream features)
    config_path = os.path.join(MODEL_FOLDER, "upstream_features_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Upstream config JSON not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    top_guids = config_data.get("top_guids", [])
    top_names = config_data.get("top_names", [])
    guid_to_index = {g: i for i, g in enumerate(top_guids)}
    name_to_index = {n: i for i, n in enumerate(top_names)}

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded top GUIDs ({len(top_guids)}) and top Names ({len(top_names)})")

    # 4) Load test JSON data
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loading test data from {TEST_JSON_FOLDER}")
    components = load_json_files(TEST_JSON_FOLDER, file_limit=EVAL_FILE_LIMIT)
    if not components:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No components found in test JSON. Exiting.")
        return

    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # Filter: keep only comps that produce at least one OUTPUT
    filtered = [
        c for c in components
        if any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))
    ]
    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Total components loaded: {len(components)}")
    if not filtered:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid comps remain after filter. Exiting.")
        return

    # 5) Build dataset rows (with progress bar)
    from tqdm import tqdm
    data_rows = []
    for comp in tqdm(filtered, desc=f"{Fore.CYAN}Building dataset rows{Style.RESET_ALL}", unit="comp"):
        cid = comp.get("Id")
        if not cid:
            continue

        ups = get_upstream_ids(comp, out_map)
        if not ups:
            continue

        downs = get_downstream_ids(comp, in_map)
        if not downs:
            continue

        # Build multi-hot vectors for upstream guids/names
        vec_guids = np.zeros(len(top_guids), dtype=int)
        vec_names = np.zeros(len(top_names), dtype=int)

        for uid in ups:
            up_comp = comp_lookup.get(uid, {})
            up_name = up_comp.get("Name", "Unknown")

            if uid in guid_to_index:
                idx_g = guid_to_index[uid]
                vec_guids[idx_g] = 1

            if up_name in name_to_index:
                idx_n = name_to_index[up_name]
                vec_names[idx_n] = 1

        feats = extract_numeric_features(comp)
        first_down_id = downs[0]
        dcomp = comp_lookup.get(first_down_id, {})
        down_label = f"{first_down_id}|{dcomp.get('Name','Unknown')}"

        row_dict = {
            "CurrentGUID": cid,
            "CurrentName": feats["Name"],
            "CurrentNumParams": feats["NumParams"],
            "CurrentNumInput": feats["NumInput"],
            "CurrentNumOutput": feats["NumOutput"],
            "DownstreamGUIDName": down_label
        }
        # Add multi-hot columns
        for i in range(len(top_guids)):
            row_dict[f"UpGUID_{i}"] = vec_guids[i]
        for i in range(len(top_names)):
            row_dict[f"UpName_{i}"] = vec_names[i]

        data_rows.append(row_dict)

    df_test = pd.DataFrame(data_rows)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Test dataset shape: {df_test.shape}")
    if df_test.empty:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No data after building rows. Exiting.")
        return

    # 6) Skip rows with unseen labels in the "cat_cols"
    for col in cat_cols:
        df_test[col] = df_test[col].astype(str)
        known_vals = set(encoders[col].classes_)
        before_count = len(df_test)
        df_test = df_test[df_test[col].isin(known_vals)]
        after_count = len(df_test)
        dropped = before_count - after_count
        if dropped > 0:
            print(
                f"{Fore.YELLOW}[INFO]{Style.RESET_ALL} Skipped {dropped} rows "
                f"because '{col}' had new/unseen labels."
            )
        # Now transform
        df_test[f"{col}_encoded"] = encoders[col].transform(df_test[col])

    # 7) Build X_test
    feature_cols = []
    # multi-hot for guids
    feature_cols += [f"UpGUID_{i}" for i in range(len(top_guids))]
    # multi-hot for names
    feature_cols += [f"UpName_{i}" for i in range(len(top_names))]
    # numeric columns
    feature_cols += [
        "CurrentGUID_encoded",
        "CurrentName_encoded",
        "CurrentNumParams",
        "CurrentNumInput",
        "CurrentNumOutput"
    ]
    X_test = df_test[feature_cols]
    y_test = df_test["DownstreamGUIDName_encoded"]

    if len(X_test) == 0:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} After skipping unseen labels, no rows remain. Exiting.")
        return

    # 8) Predict
    dtest = xgb.DMatrix(X_test)
    raw_preds = model.predict(dtest)

    if raw_preds.ndim == 1:
        # "multi:softmax"
        preds_top1 = raw_preds.astype(int)
    else:
        # "multi:softprob"
        preds_top1 = raw_preds.argmax(axis=1)

    # Evaluate
    acc = accuracy_score(y_test, preds_top1)
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Top-1 Accuracy: {acc:.4f}")

    f1_w = f1_score(y_test, preds_top1, average="weighted", zero_division=0)
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Weighted F1 Score: {f1_w:.4f}")

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Classification Report (Weighted Average):")
    print(classification_report(y_test, preds_top1, zero_division=0))

    # Show top-5 for sample
    if not X_test.empty:
        sample_idx = 0
        row = df_test.iloc[sample_idx]
        print(f"\n{Fore.MAGENTA}[Sample Row]{Style.RESET_ALL} Index {sample_idx}")
        print(f"CurrentGUID: {row['CurrentGUID']}, CurrentName: {row['CurrentName']}")

        if raw_preds.ndim > 1:
            row_probs = raw_preds[sample_idx]
            top_k_idx = row_probs.argsort()[::-1][:5]
            top_k_scores = row_probs[top_k_idx]
            down_enc = encoders["DownstreamGUIDName"]
            decoded_top = down_enc.inverse_transform(top_k_idx)
            for rank, (lbl, score) in enumerate(zip(decoded_top, top_k_scores), start=1):
                print(f"  Rank {rank} -> {lbl}, score={score:.4f}")

        actual_enc = y_test.iloc[sample_idx]
        actual_lbl = encoders["DownstreamGUIDName"].inverse_transform([actual_enc])[0]
        print(f"\n{Fore.GREEN}[ACTUAL LABEL]{Style.RESET_ALL} {actual_lbl}")

if __name__ == "__main__":
    main()
