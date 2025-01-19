import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import subprocess
from collections import Counter
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

# Shared code
from xgboost_utilities import (
    load_json_files,
    build_input_map,
    build_output_map,
    build_comp_lookup,
    get_upstream_ids,
    get_downstream_ids,
    extract_numeric_features,
    create_dated_output_subfolder
)

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"
OUTPUT_FOLDER = r"./WP_2_Vector_Database/output"
MODEL_NAME = "Full"
DEBUG_MODE = True
DEBUG_FILE_LIMIT = 1000 if DEBUG_MODE else None

# Decide how many top GUIDs and top Names to keep for multi-hot
TOP_GUIDS = 100
TOP_NAMES = 100

def main():
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loading training data from: {JSON_FOLDER}")
    components = load_json_files(JSON_FOLDER, file_limit=DEBUG_FILE_LIMIT)

    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # Filter: only comps with at least 1 OUTPUT param
    filtered = [
        c for c in components
        if any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))
    ]

    # 1) Gather all upstream GUIDs and Names
    all_up_guids = []
    all_up_names = []
    for comp in filtered:
        ups = get_upstream_ids(comp, out_map)
        for uid in ups:
            up_comp = comp_lookup.get(uid, {})
            all_up_guids.append(uid)  # the actual GUID
            all_up_names.append(up_comp.get("Name", "Unknown"))

    # 2) Get top N
    guid_counter = Counter(all_up_guids)
    name_counter = Counter(all_up_names)

    top_guids = [g for (g, freq) in guid_counter.most_common(TOP_GUIDS)]
    top_names = [n for (n, freq) in name_counter.most_common(TOP_NAMES)]

    # Build a quick map for O(1) lookups
    guid_to_index = {g: i for i, g in enumerate(top_guids)}
    name_to_index = {n: i for i, n in enumerate(top_names)}

    # Optionally, we can save these top_guids/top_names for usage in eval
    # Let’s store them in a JSON file:
    # This will help the evaluation script build the same multi-hot vectors
    config_subfolder = create_dated_output_subfolder(OUTPUT_FOLDER, MODEL_NAME)
    # We'll actually store the model here, to avoid confusion
    # remove the create_dated_output_subfolder call if you prefer a single folder.
    output_subfolder = config_subfolder

    upstream_config_path = os.path.join(output_subfolder, "upstream_features_config.json")
    with open(upstream_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "top_guids": top_guids,
            "top_names": top_names
        }, f, ensure_ascii=False, indent=2)

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Stored top GUIDs/Names config -> {upstream_config_path}")

    # 3) Build dataset rows
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

        # Create multi-hot vectors
        vec_guids = np.zeros(TOP_GUIDS, dtype=int)
        vec_names = np.zeros(TOP_NAMES, dtype=int)

        for uid in ups:
            up_comp = comp_lookup.get(uid, {})
            up_guid = uid
            up_name = up_comp.get("Name", "Unknown")

            # If up_guid is in top_guids, set that index to 1
            if up_guid in guid_to_index:
                idx_g = guid_to_index[up_guid]
                vec_guids[idx_g] = 1

            # If up_name is in top_names, set that index to 1
            if up_name in name_to_index:
                idx_n = name_to_index[up_name]
                vec_names[idx_n] = 1

        # Extract numeric features of current comp
        feats = extract_numeric_features(comp)

        # Use the first downstream as the label
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
        for i in range(TOP_GUIDS):
            row_dict[f"UpGUID_{i}"] = vec_guids[i]
        for i in range(TOP_NAMES):
            row_dict[f"UpName_{i}"] = vec_names[i]

        data_rows.append(row_dict)

    df = pd.DataFrame(data_rows)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Final dataset shape: {df.shape}")
    if df.empty:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid data found. Exiting.")
        return

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Model artifacts will be saved to: {output_subfolder}")

    # 4) Label-encode CurrentGUID, CurrentName, DownstreamGUIDName
    cat_cols = ["CurrentGUID", "CurrentName", "DownstreamGUIDName"]
    encoders = {}

    def export_encoder_to_json(encoder, output_path):
        classes_list = encoder.classes_.tolist()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(classes_list, f, ensure_ascii=False, indent=2)

    for col in cat_cols:
        df[col] = df[col].astype(str)
        enc = LabelEncoder()
        df[f"{col}_encoded"] = enc.fit_transform(df[col])
        encoders[col] = enc

        enc_path = os.path.join(output_subfolder, f"{col}_encoder.json")
        export_encoder_to_json(enc, enc_path)
        print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved encoder for {col} -> {enc_path}")

    # 5) Build feature matrix
    feature_cols = []
    # multi-hot for GUID
    feature_cols += [f"UpGUID_{i}" for i in range(TOP_GUIDS)]
    # multi-hot for Name
    feature_cols += [f"UpName_{i}" for i in range(TOP_NAMES)]

    # numeric columns + label-encoded columns
    feature_cols += [
        "CurrentGUID_encoded",
        "CurrentName_encoded",
        "CurrentNumParams",
        "CurrentNumInput",
        "CurrentNumOutput"
    ]
    X = df[feature_cols]
    y = df["DownstreamGUIDName_encoded"]

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "multi:softprob",
        "num_class": len(encoders["DownstreamGUIDName"].classes_),
        "tree_method": "hist",
        "nthread": -1
    }

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Training XGBoost model ...")
    model = xgb.train(params, dtrain, num_boost_round=30)

    model_path = os.path.join(output_subfolder, "xgboost_model.json")
    model.save_model(model_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} XGBoost model saved -> {model_path}")

    # Quick test-set check
    pred_proba = model.predict(dtest)
    preds_top1 = pred_proba.argmax(axis=1)
    acc = accuracy_score(y_test, preds_top1)
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Quick test-set Accuracy: {acc:.4f}")

    # Optionally call evaluation script automatically
    eval_script_path = os.path.join(os.path.dirname(__file__), "xgboost_evaluate.py")
    command = [
        "python",
        eval_script_path,
        "--model_folder", output_subfolder,
        "--test_json_folder", JSON_FOLDER
    ]
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Automatically calling xgboost_evaluate.py ...")
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Evaluation script failed: {str(e)}")

if __name__ == "__main__":
    main()
