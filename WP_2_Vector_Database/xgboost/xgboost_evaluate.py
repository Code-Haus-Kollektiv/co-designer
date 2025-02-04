import os
import json
import argparse
import numpy as np
import pandas as pd
import sys
import logging
import onnxruntime
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# Import shared utility functions
from xgboost_utilities import (
    load_json_files,
    build_input_map,
    build_output_map,
    build_comp_lookup,
    get_upstream_ids,
    get_downstream_ids,
    extract_numeric_features,
    load_label_encoder,
    load_features_config
)

# Import Colorama and initialize for colored outputs
from colorama import Fore, Style, init
init(autoreset=True)

# Define a custom colored formatter for logging
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

# Setup logging with colored output
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.handlers = []  # Clear previous handlers
logger.addHandler(ch)

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
EVAL_FILE_LIMIT = 50
DEFAULT_OUTPUT_FOLDER = Path("./WP_2_Vector_Database/output")
DEFAULT_JSON_FOLDER = Path("./WP_2_Vector_Database/json_chunks/Results")
FEATURES_CONFIG_FILENAME = "features_config.json"
MODEL_FILENAME = "xgboost_model.onnx"
ENCODERS_SUFFIX = "_encoder.json"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate ONNX model on test JSON data.")
    parser.add_argument("--test_json_folder", default=DEFAULT_JSON_FOLDER,
                        help="Path containing JSON test data.")
    return parser.parse_args()

def get_latest_output_folder(base_dir: Path) -> Path:
    # Get folders that end with either '_Full' or '_Debug'
    folders_full = list(base_dir.glob('*_Full'))
    folders_debug = list(base_dir.glob('*_Debug'))
    folders = folders_full + folders_debug
    if not folders:
        logger.error("No output folders found in %s", base_dir)
        sys.exit(1)
    latest = sorted(folders, key=os.path.getmtime, reverse=True)[0]
    logger.debug("Latest output folder determined: %s", latest)
    return latest

def load_label_encoders(model_folder: Path, categorical_columns):
    encoders = {}
    for col in categorical_columns:
        enc_path = model_folder / f"{col}{ENCODERS_SUFFIX}"
        if not enc_path.exists():
            logger.error("Missing encoder for %s: %s", col, enc_path)
            sys.exit(1)
        try:
            encoder = load_label_encoder(enc_path)
            # Ensure "Unknown" exists in encoder classes for handling unseen labels.
            if "Unknown" not in list(encoder.classes_):
                logger.warning("'Unknown' label not found in encoder for '%s'. Adding it.", col)
                new_classes = np.append(encoder.classes_, "Unknown")
                encoder.classes_ = new_classes
            encoders[col] = encoder
            logger.info("Loaded encoder for %s from %s", col, enc_path)
        except Exception as e:
            logger.error("Failed to load encoder for %s: %s", col, e)
            sys.exit(1)
    return encoders

def build_test_row(comp, out_map, in_map, comp_lookup, top_guids, top_names, top_input_params, use_simple_labels):
    cid = comp.get("Id")
    if not cid:
        return None
    ups = get_upstream_ids(comp, out_map)
    downs = get_downstream_ids(comp, in_map)
    if not downs:
        return None
    vec_guids = np.zeros(len(top_guids), dtype=int)
    vec_names = np.zeros(len(top_names), dtype=int)
    for uid in ups:
        up_comp = comp_lookup.get(uid, {})
        up_name = up_comp.get("Name", "Unknown")
        if uid in top_guids:
            vec_guids[top_guids.index(uid)] = 1
        if up_name in top_names:
            vec_names[top_names.index(up_name)] = 1
    vec_inp = np.zeros(len(top_input_params), dtype=int)
    for p in comp.get("Parameters", []):
        if p.get("ParameterType") == "Input" and p.get("Id"):
            pid = p["Id"]
            if pid in top_input_params:
                vec_inp[top_input_params.index(pid)] = 1
    feats = extract_numeric_features(comp)
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

def prepare_test_data(filtered_comps, out_map, in_map, comp_lookup, top_guids, top_names, top_input_params, use_simple_labels):
    data_rows = []
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(
            build_test_row, comp, out_map, in_map, comp_lookup,
            top_guids, top_names, top_input_params, use_simple_labels
        ) for comp in filtered_comps]
        for f in tqdm(futures, desc="Building test rows", unit="comp"):
            result = f.result()
            if result:
                data_rows.append(result)
    return data_rows

def main():
    args = parse_arguments()
    test_folder = Path(args.test_json_folder)
    if not test_folder.exists():
        logger.error("Test JSON folder does not exist: %s", test_folder)
        sys.exit(1)
    latest_output_folder = get_latest_output_folder(DEFAULT_OUTPUT_FOLDER)
    onnx_path = latest_output_folder / MODEL_FILENAME
    config_path = latest_output_folder / FEATURES_CONFIG_FILENAME
    if not onnx_path.exists():
        logger.error("ONNX model not found: %s", onnx_path)
        sys.exit(1)
    if not config_path.exists():
        logger.error("Feature config not found: %s", config_path)
        sys.exit(1)
    try:
        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        logger.info("Loaded ONNX model from %s", onnx_path)
    except Exception as e:
        logger.error("Failed to load ONNX model: %s", e)
        sys.exit(1)
    try:
        feature_config = load_features_config(config_path)
        logger.debug("Feature configuration loaded.")
    except Exception as e:
        logger.error("Failed to load feature configuration: %s", e)
        sys.exit(1)
    feature_cols = feature_config.get("feature_cols", [])
    top_guids = feature_config.get("top_guids", [])
    top_names = feature_config.get("top_names", [])
    top_input_params = feature_config.get("top_input_params", [])
    use_simple_labels = feature_config.get("use_simple_labels", False)
    categorical_cols = ["CurrentGUID", "CurrentName", "DownstreamGUIDName"]
    encoders = load_label_encoders(latest_output_folder, categorical_cols)
    try:
        comps_test = load_json_files(test_folder, file_limit=EVAL_FILE_LIMIT)
        if not comps_test:
            logger.error("No components found in test JSON folder: %s", test_folder)
            sys.exit(1)
    except Exception as e:
        logger.error("Failed to load test JSON files: %s", e)
        sys.exit(1)
    in_map = build_input_map(comps_test)
    out_map = build_output_map(comps_test)
    comp_lookup = build_comp_lookup(comps_test)
    filtered_comps = [c for c in comps_test if any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))]
    if not filtered_comps:
        logger.error("No valid components after filtering in test data.")
        sys.exit(1)
    logger.info("Number of components after filtering: %d", len(filtered_comps))
    test_data_rows = prepare_test_data(
        filtered_comps, out_map, in_map, comp_lookup,
        top_guids, top_names, top_input_params, use_simple_labels
    )
    df_test = pd.DataFrame(test_data_rows)
    if df_test.empty:
        logger.error("No valid test rows generated. Exiting.")
        sys.exit(1)
    logger.info("Test DataFrame shape: %s", df_test.shape)
    for col in categorical_cols:
        df_test[col] = df_test[col].astype(str)
        mapping = {cls: idx for idx, cls in enumerate(encoders[col].classes_)}
        default_idx = mapping.get("Unknown", len(mapping))
        df_test[col] = df_test[col].apply(lambda x: mapping.get(x, default_idx))
        logger.debug("Encoded column '%s' with unseen labels mapped to index %s", col, default_idx)
    df_test.rename(columns={
        "CurrentGUID": "CurrentGUID_encoded",
        "CurrentName": "CurrentName_encoded",
        "DownstreamGUIDName": "DownstreamGUIDName_encoded"
    }, inplace=True)
    logger.debug("Downstream label distribution in TEST after encoding:\n%s",
                 df_test["DownstreamGUIDName_encoded"].value_counts().head(20))
    missing_feats = set(feature_cols) - set(df_test.columns)
    if missing_feats:
        logger.warning("Missing features in test data: %s", missing_feats)
        for mf in missing_feats:
            df_test[mf] = 0
    X_test = df_test[feature_cols].astype(np.float32)
    y_test = df_test["DownstreamGUIDName_encoded"].values
    logger.debug("X_test shape: %s, y_test shape: %s", X_test.shape, y_test.shape)
    logger.debug("Number of distinct labels in y_test: %d", len(np.unique(y_test)))
    
    # Compute chance accuracy for baseline comparison
    chance_accuracy = 1 / len(np.unique(y_test)) if len(np.unique(y_test)) > 0 else 0
    logger.info("Chance accuracy (random guessing): %.4f", chance_accuracy)
    
    try:
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: X_test.to_numpy()}
        ort_outs = session.run(None, ort_inputs)
        probs = ort_outs[0]
        logger.info("Inference completed successfully.")
    except Exception as e:
        logger.error("Inference failed: %s", e)
        sys.exit(1)
    if probs.ndim == 1:
        logger.warning("Single-class output detected. Assigning all predictions to class 0.")
        preds = np.zeros_like(probs, dtype=int)
    else:
        preds = probs.argmax(axis=1)
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
    logger.info("[RESULT] Accuracy: %.4f", acc)
    logger.info("[RESULT] Weighted F1 Score: %.4f", f1_w)
    logger.info("Classification Report:\n%s", classification_report(y_test, preds, zero_division=0))
    unique_test_labels = np.unique(y_test)
    unique_pred_labels = np.unique(preds)
    if len(unique_test_labels) > 1 and len(unique_pred_labels) > 1:
        cm = confusion_matrix(y_test, preds)
        logger.debug("Confusion Matrix:\n%s", cm)
    else:
        logger.debug("Confusion matrix not displayed (single class in y_test or preds).")
    if len(X_test) > 0 and probs.ndim == 2:
        row0 = probs[0]
        top5_idx = row0.argsort()[::-1][:5]
        top5_scores = row0[top5_idx]
        actual_enc = y_test[0]
        try:
            actual_lbl = encoders["DownstreamGUIDName"].inverse_transform([actual_enc])[0]
        except Exception as e:
            logger.error("Failed to inverse transform label '%s': %s", actual_enc, e)
            actual_lbl = "Unknown"
        logger.debug("Sample #0 Actual DownstreamGUIDName: %s", actual_lbl)
        for rank, (idx_c, sc) in enumerate(zip(top5_idx, top5_scores), start=1):
            try:
                lbl_str = encoders["DownstreamGUIDName"].inverse_transform([idx_c])[0]
            except Exception as e:
                logger.error("Failed to inverse transform index '%s': %s", idx_c, e)
                lbl_str = "Unknown"
            print(f"  Rank {rank}: {lbl_str}, score={sc:.4f}")
    else:
        logger.info("Top-5 predictions not available (single class or empty X_test).")
    
    # Do not re-save the ONNX model or label map here.
    logger.info("Evaluation complete. Metrics reported above.")

if __name__ == "__main__":
    main()
