#!/usr/bin/env python
"""
train.py

Trains an XGBClassifier to predict "DownstreamGUIDName" from Grasshopper data.
Exports:
  1) xgboost_model.onnx  (the trained model in ONNX format)
  2) index_to_label.json (map from integer class -> "GUID|Name" string)
  3) DownstreamGUIDName_encoder.json, CurrentGUID_encoder.json, CurrentName_encoder.json
  4) features_config.json (feature configuration)
  
Usage:
  python train.py
"""
import os
import json
import numpy as np
import pandas as pd
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

import colorama
from colorama import Fore, Style
colorama.init(strip=False, convert=True)

import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.utils import save_model as save_onnx_model
from onnxmltools.convert.common.data_types import FloatTensorType

# Your utility code for reading JSON chunks, building maps, etc.
from xgboost_utilities import (
    load_json_files,
    build_input_map,
    build_output_map,
    build_comp_lookup,
    get_upstream_ids,
    get_downstream_ids,
    extract_numeric_features,
    create_dated_output_subfolder,
    save_label_encoder,
    save_features_config
)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"

OUTPUT_FOLDER = r"./WP_2_Vector_Database/output"
MODEL_NAME = "Full"

DEBUG_MODE = False
DEBUG_FILE_LIMIT = 2000

MIN_FREQ = 1
TOP_GUIDS = 100
TOP_NAMES = 100
TOP_INPUTS = 100
USE_SIMPLE_LABELS = False

FEATURES_CONFIG_FILENAME = "features_config.json"
ENCODERS_SUFFIX = "_encoder.json"

def process_component(comp, out_map, in_map, comp_lookup,
                      top_guids, top_names, top_input_params,
                      guid_to_index, name_to_index, input_param_to_index,
                      use_simple_labels):
    """
    Build features for one "current" component => next (downstream) label.
    """
    cid = comp.get("Id")
    if not cid:
        return None

    ups = get_upstream_ids(comp, out_map)
    downs = get_downstream_ids(comp, in_map)
    if not downs:
        return None

    # One-hot for top upstream GUIDs + Names
    vec_guids = np.zeros(len(top_guids), dtype=int)
    vec_names = np.zeros(len(top_names), dtype=int)
    for uid in ups:
        up_comp = comp_lookup.get(uid, {})
        up_name = up_comp.get("Name", "Unknown")
        if uid in guid_to_index:
            vec_guids[guid_to_index[uid]] = 1
        if up_name in name_to_index:
            vec_names[name_to_index[up_name]] = 1

    # One-hot for top input param IDs
    vec_inp = np.zeros(len(top_input_params), dtype=int)
    for p in comp.get("Parameters", []):
        if p.get("ParameterType") == "Input" and p.get("Id"):
            pid = p["Id"]
            if pid in input_param_to_index:
                vec_inp[input_param_to_index[pid]] = 1

    # Basic numeric features
    feats = extract_numeric_features(comp)

    # Next (downstream) label
    first_down_id = downs[0]
    dcomp = comp_lookup.get(first_down_id, {})
    if use_simple_labels:
        target_label = dcomp.get("Name", "Unknown")
    else:
        target_label = f"{first_down_id}|{dcomp.get('Name','Unknown')}"

    row = {
        "CurrentGUID": cid,
        "CurrentName": feats["Name"],
        "CurrentNumParams": feats["NumParams"],
        "CurrentNumInput": feats["NumInput"],
        "CurrentNumOutput": feats["NumOutput"],
        "DownstreamGUIDName": target_label
    }
    for i in range(len(top_guids)):
        row[f"UpGUID_{i}"] = vec_guids[i]
    for i in range(len(top_names)):
        row[f"UpName_{i}"] = vec_names[i]
    for i in range(len(top_input_params)):
        row[f"InpParam_{i}"] = vec_inp[i]

    return row

def build_rows_parallel(components, out_map, in_map, comp_lookup,
                        top_guids, top_names, top_input_params,
                        guid_to_index, name_to_index, input_param_to_index,
                        use_simple_labels):
    """
    Build data rows in parallel for all 'components'.
    """
    from tqdm import tqdm
    rows = []
    with ProcessPoolExecutor() as executor:
        futures = []
        for comp in components:
            futures.append(executor.submit(
                process_component,
                comp, out_map, in_map, comp_lookup,
                top_guids, top_names, top_input_params,
                guid_to_index, name_to_index, input_param_to_index,
                use_simple_labels
            ))
        for f in tqdm(futures, desc=f"{Fore.CYAN}Building rows{Style.RESET_ALL}", unit="comp"):
            result = f.result()
            if result is not None:
                rows.append(result)
    return rows


def main():
    # --------------------------------------------------------------------
    # 1) Load JSON, build forward/backward maps
    # --------------------------------------------------------------------
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loading JSON from: {Fore.YELLOW}{JSON_FOLDER}{Style.RESET_ALL}")
    file_limit = None  # Set to DEBUG_FILE_LIMIT if needed
    if DEBUG_MODE:
        file_limit = DEBUG_FILE_LIMIT
    components = load_json_files(JSON_FOLDER, file_limit=file_limit)

    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # Filter => only components with an output parameter
    filtered = [c for c in components if any(
        p.get("ParameterType") == "Output" for p in c.get("Parameters", [])
    )]
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} # components with outputs = {len(filtered)}")

    # --------------------------------------------------------------------
    # 2) Gather top upstream GUIDs/Names
    # --------------------------------------------------------------------
    all_up_guids, all_up_names = [], []
    for comp in filtered:
        ups = get_upstream_ids(comp, out_map)
        for uid in ups:
            up_comp = comp_lookup.get(uid, {})
            all_up_guids.append(uid)
            all_up_names.append(up_comp.get("Name", "Unknown"))

    guid_counter = Counter(all_up_guids)
    name_counter = Counter(all_up_names)
    top_guids_list = [g for (g, _) in guid_counter.most_common(TOP_GUIDS)]
    top_names_list = [n for (n, _) in name_counter.most_common(TOP_NAMES)]
    guid_to_index = {g: i for i, g in enumerate(top_guids_list)}
    name_to_index = {n: i for i, n in enumerate(top_names_list)}

    # Top input param IDs
    input_param_counter = Counter()
    for comp in filtered:
        for p in comp.get("Parameters", []):
            if p.get("ParameterType") == "Input" and p.get("Id"):
                input_param_counter[p["Id"]] += 1
    top_input_params_list = [pid for (pid, _) in input_param_counter.most_common(TOP_INPUTS)]
    input_param_to_index = {pid: i for i, pid in enumerate(top_input_params_list)}

    # Create output subfolder
    output_subfolder = os.path.join(OUTPUT_FOLDER, f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_1_Full")
    os.makedirs(output_subfolder, exist_ok=True)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Output folder: {Fore.YELLOW}{output_subfolder}{Style.RESET_ALL}")

    # --------------------------------------------------------------------
    # 3) Build dataset in parallel
    # --------------------------------------------------------------------
    data_rows = build_rows_parallel(
        filtered,
        out_map,
        in_map,
        comp_lookup,
        top_guids_list,
        top_names_list,
        top_input_params_list,
        guid_to_index,
        name_to_index,
        input_param_to_index,
        USE_SIMPLE_LABELS
    )
    # Filter out None values
    data_rows = [row for row in data_rows if row is not None]
    df = pd.DataFrame(data_rows)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Final dataset shape:", df.shape)

    # --------------------------------------------------------------------
    # 4) Merge rare labels into "Unknown" and drop singletons
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Downstream label distribution BEFORE merges:")
    print(df["DownstreamGUIDName"].value_counts())

    # Merge rare labels into "Unknown"
    label_freq = df["DownstreamGUIDName"].value_counts()
    rare_labels = label_freq[label_freq < MIN_FREQ].index
    df.loc[df["DownstreamGUIDName"].isin(rare_labels), "DownstreamGUIDName"] = "Unknown"

    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Downstream label distribution AFTER merges:")
    print(df["DownstreamGUIDName"].value_counts())

    # Drop singletons entirely
    label_counts = Counter(df["DownstreamGUIDName"])
    singletons = [lbl for lbl, cnt in label_counts.items() if cnt < 2]
    if singletons:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Dropping single-sample classes:", singletons)
        df = df[~df["DownstreamGUIDName"].isin(singletons)]

    # --------------------------------------------------------------------
    # 5) Encode 'CurrentGUID' and 'CurrentName' in df
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Encoding 'CurrentGUID' and 'CurrentName'...")

    # Check if 'CurrentGUID' and 'CurrentName' exist in df
    if 'CurrentGUID' not in df.columns or 'CurrentName' not in df.columns:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} 'CurrentGUID' or 'CurrentName' columns are missing from DataFrame.")
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Available columns: {df.columns.tolist()}")
        return

    # Initialize LabelEncoders for features
    le_guid = LabelEncoder()
    le_name = LabelEncoder()

    # Encode 'CurrentGUID'
    df["CurrentGUID_encoded"] = le_guid.fit_transform(df["CurrentGUID"].astype(str))
    # Encode 'CurrentName'
    df["CurrentName_encoded"] = le_name.fit_transform(df["CurrentName"].astype(str))

    # --------------------------------------------------------------------
    # 6) Define Feature Columns Final
    # --------------------------------------------------------------------
    feature_cols_final = (
        ["CurrentNumParams", "CurrentNumInput", "CurrentNumOutput"] +
        [f"UpGUID_{i}" for i in range(len(top_guids_list))] +
        [f"UpName_{i}" for i in range(len(top_names_list))] +
        [f"InpParam_{i}" for i in range(len(top_input_params_list))] +
        ["CurrentGUID_encoded", "CurrentName_encoded"]
    )
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Feature Columns Final: {feature_cols_final}")

    # --------------------------------------------------------------------
    # 7) Verify All Feature Columns Exist in DataFrame
    # --------------------------------------------------------------------
    missing_features = set(feature_cols_final) - set(df.columns)
    if missing_features:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Missing features in DataFrame: {missing_features}")
        return
    else:
        print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} All feature columns are present in DataFrame.")

    # --------------------------------------------------------------------
    # 8) Select X and y
    # --------------------------------------------------------------------
    X = df[feature_cols_final]
    y = df["DownstreamGUIDName"]

    # --------------------------------------------------------------------
    # 9) Split the Data (Stratified)
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    # --------------------------------------------------------------------
    # 10) Encode Labels
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Encoding labels...")
    le_label = LabelEncoder()
    y_train_encoded = le_label.fit_transform(y_train)
    y_val_encoded = le_label.transform(y_val)

    # --------------------------------------------------------------------
    # 11) Save Encoders and Feature Configuration
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Saving encoders and feature configuration...")

    # Save Label Encoders
    encoder_paths = {
        "DownstreamGUIDName": os.path.join(output_subfolder, "DownstreamGUIDName_encoder.json"),
        "CurrentGUID": os.path.join(output_subfolder, "CurrentGUID_encoder.json"),
        "CurrentName": os.path.join(output_subfolder, "CurrentName_encoder.json"),
    }

    save_label_encoder(le_label, encoder_paths["DownstreamGUIDName"])
    save_label_encoder(le_guid, encoder_paths["CurrentGUID"])
    save_label_encoder(le_name, encoder_paths["CurrentName"])

    # Save Feature Configuration
    feature_config = {
        "feature_cols": feature_cols_final,
        "top_guids": top_guids_list,
        "top_names": top_names_list,
        "top_input_params": top_input_params_list,
        "use_simple_labels": USE_SIMPLE_LABELS
    }

    features_config_path = os.path.join(output_subfolder, FEATURES_CONFIG_FILENAME)
    save_features_config(feature_config, features_config_path)

    # --------------------------------------------------------------------
    # 12) Train the XGBoost Classifier
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Training the XGBoost classifier...")
    unique_labels_train = np.unique(y_train_encoded)
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} # unique labels in training after splits = {len(unique_labels_train)}")
    if len(unique_labels_train) < 2:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Only one class in training set => not truly multi-class")

    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(unique_labels_train),
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist",  # Use GPU-accelerated tree construction if available
        n_jobs=-1,               # Use all available CPU cores for data loading
        reg_lambda=1.0,
        alpha=0.0
    )

    xgb_clf.fit(
        X_train,
        y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        early_stopping_rounds=20,
        verbose=True
    )

    # --------------------------------------------------------------------
    # 13) Evaluation
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Evaluating the model on validation set...")
    preds_val = xgb_clf.predict(X_val)
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Classification report on validation set:")
    print(classification_report(y_val_encoded, preds_val, zero_division=0))

    # --------------------------------------------------------------------
    # 14) Convert XGBoost Model to ONNX
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Converting the XGBoost model to ONNX format...")
    booster = xgb_clf.get_booster()
    booster.feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    onnx_path = os.path.join(output_subfolder, "xgboost_model.onnx")
    initial_types = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_types)
    save_onnx_model(onnx_model, onnx_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} ONNX model saved -> {onnx_path}")

    # --------------------------------------------------------------------
    # 15) Save the Index-to-Label Map
    # --------------------------------------------------------------------
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Saving the index-to-label map...")
    index_to_label = {i: lbl_str for i, lbl_str in enumerate(le_label.classes_)}

    map_path = os.path.join(output_subfolder, "index_to_label.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved index_to_label map -> {map_path}")

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Training complete. Model + maps written to: {output_subfolder}")


if __name__ == "__main__":
    main()
