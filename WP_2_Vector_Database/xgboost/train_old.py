#!/usr/bin/env python
"""
train.py

Trains an XGBClassifier to predict "DownstreamGUIDName" from Grasshopper data.
Exports:
  1) xgboost_model.onnx  (the trained model in ONNX format)
  2) index_to_label.json (map from integer class -> "GUID|Name" string)
  3) DownstreamGUIDName_encoder.json, etc. (optional)

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
    create_dated_output_subfolder
)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
JSON_FOLDER = r"./WP_2_Vector_Database/json_chunks/Results"

OUTPUT_FOLDER = r"./WP_2_Vector_Database/output"
MODEL_NAME = "Full"

DEBUG_MODE = False
DEBUG_FILE_LIMIT = 500

MIN_FREQ = 1
TOP_GUIDS = 100
TOP_NAMES = 100
TOP_INPUTS = 100
USE_SIMPLE_LABELS = False

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
    file_limit = DEBUG_FILE_LIMIT if DEBUG_MODE else None
    components = load_json_files(JSON_FOLDER, file_limit=file_limit)

    in_map = build_input_map(components)
    out_map = build_output_map(components)
    comp_lookup = build_comp_lookup(components)

    # Filter => only comps with an output param
    filtered = [c for c in components if any(
        p.get("ParameterType") == "Output" for p in c.get("Parameters", [])
    )]
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} # components with outputs = {len(filtered)}")

    # --------------------------------------------------------------------
    # 2) Gather top upstream guids/names
    # --------------------------------------------------------------------
    all_up_guids, all_up_names = [], []
    for comp in filtered:
        ups = get_upstream_ids(comp, out_map)
        for uid in ups:
            up_comp = comp_lookup.get(uid, {})
            all_up_guids.append(uid)
            all_up_names.append(up_comp.get("Name", "Unknown"))

    import collections
    guid_counter = collections.Counter(all_up_guids)
    name_counter = collections.Counter(all_up_names)
    top_guids_list = [g for (g, _) in guid_counter.most_common(TOP_GUIDS)]
    top_names_list = [n for (n, _) in name_counter.most_common(TOP_NAMES)]
    guid_to_index = {g: i for i, g in enumerate(top_guids_list)}
    name_to_index = {n: i for i, n in enumerate(top_names_list)}

    # top input param IDs
    input_param_counter = collections.Counter()
    for comp in filtered:
        for p in comp.get("Parameters", []):
            if p.get("ParameterType") == "Input" and p.get("Id"):
                input_param_counter[p["Id"]] += 1
    top_input_params_list = [pid for (pid, _) in input_param_counter.most_common(TOP_INPUTS)]
    input_param_to_index = {pid: i for i, pid in enumerate(top_input_params_list)}

    # Create output subfolder
    output_subfolder = create_dated_output_subfolder(OUTPUT_FOLDER, MODEL_NAME)
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
    df = pd.DataFrame(data_rows)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Final dataset shape:", df.shape)

    # Debug distribution
    label_dist_before = df["DownstreamGUIDName"].value_counts()
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Downstream label distribution BEFORE merges:\n",
          label_dist_before.head(20))

    # Merge rare => "Unknown"
    label_freq = df["DownstreamGUIDName"].value_counts()
    rare_labels = label_freq[label_freq < MIN_FREQ].index
    df.loc[df["DownstreamGUIDName"].isin(rare_labels), "DownstreamGUIDName"] = "Unknown"

    label_dist_after = df["DownstreamGUIDName"].value_counts()
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Downstream label distribution AFTER merges:\n",
          label_dist_after.head(20))

    # Drop singletons entirely
    from collections import Counter
    label_counts = Counter(df["DownstreamGUIDName"])
    singletons = [lbl for lbl, cnt in label_counts.items() if cnt < 2]
    if singletons:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Dropping single-sample classes:", singletons)
        df = df[~df["DownstreamGUIDName"].isin(singletons)]

    # --------------------------------------------------------------------
    # 4) Re-fit label encoder => contiguous 0..k-1
    # --------------------------------------------------------------------
    df["DownstreamGUIDName"] = df["DownstreamGUIDName"].astype(str)
    le_label = LabelEncoder()
    df["DownstreamGUIDName_encoded"] = le_label.fit_transform(df["DownstreamGUIDName"])

    # Also re-encode CurrentGUID, CurrentName if you want them as features
    df["CurrentGUID"] = df["CurrentGUID"].astype(str)
    df["CurrentName"] = df["CurrentName"].astype(str)
    le_guid = LabelEncoder()
    df["CurrentGUID_encoded"] = le_guid.fit_transform(df["CurrentGUID"])
    le_name = LabelEncoder()
    df["CurrentName_encoded"] = le_name.fit_transform(df["CurrentName"])

    # Drop original string columns
    df.drop(columns=["CurrentGUID", "CurrentName", "DownstreamGUIDName"], inplace=True)

    # Build final feature list
    feature_cols = []
    feature_cols += ["CurrentNumParams", "CurrentNumInput", "CurrentNumOutput"]
    feature_cols += [f"UpGUID_{i}" for i in range(len(top_guids_list))]
    feature_cols += [f"UpName_{i}" for i in range(len(top_names_list))]
    feature_cols += [f"InpParam_{i}" for i in range(len(top_input_params_list))]
    # Then our 2 re-encoded columns
    feature_cols += ["CurrentGUID_encoded", "CurrentName_encoded"]

    X = df[feature_cols]
    y = df["DownstreamGUIDName_encoded"]

    unique_labels = np.unique(y)
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} # unique labels in training after merges = {len(unique_labels)}")
    if len(unique_labels) < 2:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Only one class => not truly multi-class")

    # --------------------------------------------------------------------
    # 5) Train
    # --------------------------------------------------------------------
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")

    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(unique_labels),
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist",          # Use GPU-accelerated tree construction
        n_jobs=-1,                       # Use all available CPU cores for data loading
        reg_lambda=1.0,
        alpha=0.0
    )

    xgb_clf.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=20,
                verbose=True)

    # Quick check on validation
    preds_val = xgb_clf.predict(X_val)
    print(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Classification report on validation set:")
    print(classification_report(y_val, preds_val, zero_division=0))

    # --------------------------------------------------------------------
    # 6) Convert XGBoost => ONNX
    # --------------------------------------------------------------------
    booster = xgb_clf.get_booster()
    booster.feature_names = [f"f{i}" for i in range(X_train.shape[1])]

    onnx_path = os.path.join(output_subfolder, "xgboost_model.onnx")
    initial_types = [("input", FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_xgboost(booster, initial_types=initial_types)
    save_onnx_model(onnx_model, onnx_path)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} ONNX model saved -> {onnx_path}")

    # --------------------------------------------------------------------
    # 7) Save the index->label map => "index_to_label.json"
    # --------------------------------------------------------------------
    index_to_label = {}
    for i, lbl_str in enumerate(le_label.classes_):
        index_to_label[i] = lbl_str  # e.g. "guid|Name" or "Unknown"

    map_path = os.path.join(output_subfolder, "index_to_label.json")
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, indent=2)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved index_to_label map -> {map_path}")

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Training complete. Model + map written to: {output_subfolder}")

if __name__ == "__main__":
    main()
