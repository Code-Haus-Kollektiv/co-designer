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

# Optional: check for GPU availability (if using gpu_hist)
try:
    import GPUtil
except ImportError:
    GPUtil = None

# Import Colorama and initialize (for colored outputs)
from colorama import Fore, Style, init
init(autoreset=True)

import logging

# Define a custom logging formatter with color support
class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        record.msg = f"{color}{record.msg}{Style.RESET_ALL}"
        return super().format(record)

# Set up logger with colored output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.handlers = []  # Clear any existing handlers
logger.addHandler(ch)

# Import ONNX conversion utilities
import onnxmltools
from onnxmltools.convert import convert_xgboost
from onnxmltools.utils import save_model as save_onnx_model
from onnxmltools.convert.common.data_types import FloatTensorType

# Import shared utility functions
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

# ----------------------------
# CONFIGURATION
# ----------------------------
CONFIG = {
    "JSON_FOLDER": r"./WP_2_Vector_Database/json_chunks/Results",
    "OUTPUT_FOLDER": r"./WP_2_Vector_Database/output",
    "DEBUG_MODE": False,
    "DEBUG_FILE_LIMIT": 500,
    "MIN_FREQ": 1,        # Consider increasing if you wish to merge rare classes
    "TOP_GUIDS": 100,
    "TOP_NAMES": 100,
    "TOP_INPUTS": 100,
    "USE_SIMPLE_LABELS": False,
    "FEATURES_CONFIG_FILENAME": "features_config.json",
    "ENCODERS_SUFFIX": "_encoder.json"
}

def check_gpu_availability():
    if GPUtil is not None:
        gpus = GPUtil.getGPUs()
        if gpus:
            logger.info("GPU(s) available: " + ", ".join([gpu.name for gpu in gpus]))
            return True
    logger.info("No GPU detected. Falling back to CPU tree method.")
    return False

def process_component(comp, out_map, in_map, comp_lookup,
                      top_guids, top_names, top_input_params,
                      guid_to_index, name_to_index, input_param_to_index,
                      use_simple_labels):
    """Build features for one component."""
    cid = comp.get("Id")
    if not cid:
        return None

    ups = get_upstream_ids(comp, out_map)
    downs = get_downstream_ids(comp, in_map)
    if not downs:
        return None

    # One-hot encoding for upstream GUIDs and Names
    vec_guids = np.zeros(len(top_guids), dtype=int)
    vec_names = np.zeros(len(top_names), dtype=int)
    for uid in ups:
        up_comp = comp_lookup.get(uid, {})
        up_name = up_comp.get("Name", "Unknown")
        if uid in guid_to_index:
            vec_guids[guid_to_index[uid]] = 1
        if up_name in name_to_index:
            vec_names[name_to_index[up_name]] = 1

    # One-hot encoding for top input parameters
    vec_inp = np.zeros(len(top_input_params), dtype=int)
    for p in comp.get("Parameters", []):
        if p.get("ParameterType") == "Input" and p.get("Id"):
            pid = p["Id"]
            if pid in input_param_to_index:
                vec_inp[input_param_to_index[pid]] = 1

    feats = extract_numeric_features(comp)

    # Target label: possibility to aggregate multiple downstream components
    first_down_id = downs[0]
    dcomp = comp_lookup.get(first_down_id, {})
    if use_simple_labels:
        target_label = dcomp.get("Name", "Unknown")
    else:
        target_label = f"{first_down_id}|{dcomp.get('Name', 'Unknown')}"

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
    """Build dataset rows in parallel using ProcessPoolExecutor."""
    rows = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_component,
                comp, out_map, in_map, comp_lookup,
                top_guids, top_names, top_input_params,
                guid_to_index, name_to_index, input_param_to_index,
                use_simple_labels
            ) for comp in components
        ]
        for f in futures:
            result = f.result()
            if result is not None:
                rows.append(result)
    return rows

def main():
    # Optionally adjust GPU usage based on availability
    use_gpu = check_gpu_availability()
    logger.info("Loading JSON from: %s", CONFIG["JSON_FOLDER"])
    file_limit = CONFIG["DEBUG_FILE_LIMIT"] if CONFIG["DEBUG_MODE"] else None
    components = load_json_files(CONFIG["JSON_FOLDER"], file_limit=file_limit)

    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # Filter components with at least one output parameter.
    filtered = [c for c in components if any(
        p.get("ParameterType") == "Output" for p in c.get("Parameters", [])
    )]
    logger.info("# components with outputs = %d", len(filtered))

    # Gather top upstream GUIDs/Names
    all_up_guids, all_up_names = [], []
    for comp in filtered:
        ups = get_upstream_ids(comp, out_map)
        for uid in ups:
            up_comp = comp_lookup.get(uid, {})
            all_up_guids.append(uid)
            all_up_names.append(up_comp.get("Name", "Unknown"))

    guid_counter = Counter(all_up_guids)
    name_counter = Counter(all_up_names)
    top_guids_list = [g for (g, _) in guid_counter.most_common(CONFIG["TOP_GUIDS"])]
    top_names_list = [n for (n, _) in name_counter.most_common(CONFIG["TOP_NAMES"])]
    guid_to_index = {g: i for i, g in enumerate(top_guids_list)}
    name_to_index = {n: i for i, n in enumerate(top_names_list)}

    # Top input param IDs
    input_param_counter = Counter()
    for comp in filtered:
        for p in comp.get("Parameters", []):
            if p.get("ParameterType") == "Input" and p.get("Id"):
                input_param_counter[p["Id"]] += 1
    top_input_params_list = [pid for (pid, _) in input_param_counter.most_common(CONFIG["TOP_INPUTS"])]
    input_param_to_index = {pid: i for i, pid in enumerate(top_input_params_list)}

    # Create a dated output subfolder
    model_name = "Debug" if CONFIG["DEBUG_MODE"] else "Full"
    output_subfolder = create_dated_output_subfolder(CONFIG["OUTPUT_FOLDER"], model_name)
    logger.info("Output folder: %s", output_subfolder)

    # Build dataset in parallel
    data_rows = build_rows_parallel(
        filtered, out_map, in_map, comp_lookup,
        top_guids_list, top_names_list, top_input_params_list,
        guid_to_index, name_to_index, input_param_to_index,
        CONFIG["USE_SIMPLE_LABELS"]
    )
    data_rows = [row for row in data_rows if row is not None]
    df = pd.DataFrame(data_rows)
    logger.info("Final dataset shape: %s", df.shape)

    # Merge rare labels and drop singletons
    logger.debug("Downstream label distribution BEFORE merge:\n%s", df["DownstreamGUIDName"].value_counts())
    label_freq = df["DownstreamGUIDName"].value_counts()
    rare_labels = label_freq[label_freq < CONFIG["MIN_FREQ"]].index
    df.loc[df["DownstreamGUIDName"].isin(rare_labels), "DownstreamGUIDName"] = "Unknown"
    logger.debug("Downstream label distribution AFTER merge:\n%s", df["DownstreamGUIDName"].value_counts())
    singletons = [lbl for lbl, cnt in Counter(df["DownstreamGUIDName"]).items() if cnt < 2]
    if singletons:
        logger.warning("Dropping single-sample classes: %s", singletons)
        df = df[~df["DownstreamGUIDName"].isin(singletons)]

    # Encode 'CurrentGUID' and 'CurrentName'
    logger.debug("Encoding 'CurrentGUID' and 'CurrentName'...")
    if 'CurrentGUID' not in df.columns or 'CurrentName' not in df.columns:
        logger.error("'CurrentGUID' or 'CurrentName' missing from DataFrame: %s", df.columns.tolist())
        return

    le_guid = LabelEncoder()
    le_name = LabelEncoder()
    df["CurrentGUID_encoded"] = le_guid.fit_transform(df["CurrentGUID"].astype(str))
    df["CurrentName_encoded"] = le_name.fit_transform(df["CurrentName"].astype(str))

    # Define final feature columns
    feature_cols_final = (
        ["CurrentNumParams", "CurrentNumInput", "CurrentNumOutput"] +
        [f"UpGUID_{i}" for i in range(len(top_guids_list))] +
        [f"UpName_{i}" for i in range(len(top_names_list))] +
        [f"InpParam_{i}" for i in range(len(top_input_params_list))] +
        ["CurrentGUID_encoded", "CurrentName_encoded"]
    )
    logger.debug("Feature Columns Final: %s", feature_cols_final)
    missing_features = set(feature_cols_final) - set(df.columns)
    if missing_features:
        logger.error("Missing features in DataFrame: %s", missing_features)
        return
    else:
        logger.info("All feature columns are present in DataFrame.")

    X = df[feature_cols_final]
    y = df["DownstreamGUIDName"]

    # Stratified split
    logger.debug("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y
    )
    logger.debug("X_train shape: %s, X_val shape: %s", X_train.shape, X_val.shape)

    # Encode labels
    logger.debug("Encoding labels...")
    le_label = LabelEncoder()
    y_train_encoded = le_label.fit_transform(y_train)
    y_val_encoded = le_label.transform(y_val)

    # Save encoders and feature config
    encoder_paths = {
        "DownstreamGUIDName": os.path.join(output_subfolder, "DownstreamGUIDName_encoder.json"),
        "CurrentGUID": os.path.join(output_subfolder, "CurrentGUID_encoder.json"),
        "CurrentName": os.path.join(output_subfolder, "CurrentName_encoder.json"),
    }
    save_label_encoder(le_label, encoder_paths["DownstreamGUIDName"])
    save_label_encoder(le_guid, encoder_paths["CurrentGUID"])
    save_label_encoder(le_name, encoder_paths["CurrentName"])

    feature_config = {
        "feature_cols": feature_cols_final,
        "top_guids": top_guids_list,
        "top_names": top_names_list,
        "top_input_params": top_input_params_list,
        "use_simple_labels": CONFIG["USE_SIMPLE_LABELS"]
    }
    features_config_path = os.path.join(output_subfolder, CONFIG["FEATURES_CONFIG_FILENAME"])
    save_features_config(feature_config, features_config_path)

    # Train XGBoost classifier
    logger.debug("Training the XGBoost classifier...")
    unique_labels_train = np.unique(y_train_encoded)
    logger.debug("# unique labels in training = %d", len(unique_labels_train))
    if len(unique_labels_train) < 2:
        logger.warning("Only one class in training set; not truly multi-class")

    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist" if use_gpu else "hist",
        n_jobs=-1,
        reg_lambda=1.0,
        alpha=0.0
    )
    xgb_clf.fit(
        X_train, y_train_encoded,
        eval_set=[(X_val, y_val_encoded)],
        early_stopping_rounds=20,
        verbose=True
    )

    # Evaluation on validation set
    logger.debug("Evaluating the model on validation set...")
    preds_val = xgb_clf.predict(X_val)
    logger.debug("Classification report on validation set:\n%s", classification_report(y_val_encoded, preds_val, zero_division=0))

    # Convert model to ONNX
    logger.debug("Converting XGBoost model to ONNX format...")
    booster = xgb_clf.get_booster()
    booster.feature_names = [f"f{i}" for i in range(X_train.shape[1])]
    onnx_path = os.path.join(output_subfolder, "xgboost_model.onnx")
    initial_types = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_types)
    save_onnx_model(onnx_model, onnx_path)
    logger.info("ONNX model saved -> %s", onnx_path)

    # Save index-to-label map
    logger.debug("Saving index-to-label map...")
    index_to_label = {i: lbl for i, lbl in enumerate(le_label.classes_)}
    map_path = os.path.join(output_subfolder, "index_to_label.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2)
    logger.info("Saved index_to_label map -> %s", map_path)
    logger.info("Training complete. Model and maps written to: %s", output_subfolder)

if __name__ == "__main__":
    main()
