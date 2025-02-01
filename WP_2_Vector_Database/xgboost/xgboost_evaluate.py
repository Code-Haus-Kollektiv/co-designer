import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import onnxruntime

import colorama
from colorama import Fore, Style
colorama.init(strip=False, convert=True)

from tqdm import tqdm

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

from pathlib import Path
import glob
import logging
import sys
import time
from typing import List, Dict, Any


# -----------------------------
# Configuration Parameters
# -----------------------------

# Constants
EVAL_FILE_LIMIT = 50
DEFAULT_OUTPUT_FOLDER = Path("./WP_2_Vector_Database/output")
DEFAULT_JSON_FOLDER = Path("./WP_2_Vector_Database/json_chunks/Results")
FEATURES_CONFIG_FILENAME = "features_config.json"
MODEL_FILENAME = "xgboost_model.onnx"
ENCODERS_SUFFIX = "_encoder.json"

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'DEBUG'

# -----------------------------
# End of Configuration
# -----------------------------


class ColorFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors based on the log level.
    """

    # Define color codes for different log levels
    COLOR_CODES = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA + Style.BRIGHT,
    }

    def format(self, record):
        color = self.COLOR_CODES.get(record.levelno, Fore.WHITE)
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to lowest level to allow all messages through

# Create console handler and set level to DEBUG
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

# Create and set custom formatter
formatter = ColorFormatter(
    fmt='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
ch.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(ch)

# Update logging level based on configuration
numeric_level = getattr(logging, LOG_LEVEL.upper(), None)
if not isinstance(numeric_level, int):
    logger.error(f"Invalid log level: {LOG_LEVEL}. Defaulting to INFO.")
    numeric_level = logging.INFO
logger.setLevel(numeric_level)
logger.debug("Logging level set.")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ONNX model on test JSON data.")
    parser.add_argument(
        "--test_json_folder",
        required=False,
        default=DEFAULT_JSON_FOLDER,
        help="Path containing JSON test data."
    )
    return parser.parse_args()


def get_latest_output_folder(base_dir: Path) -> Path:
    """
    Retrieves the latest output folder based on modification time.
    Assumes that output folders end with '_Full'.
    """
    list_of_folders = sorted(
        base_dir.glob('*_Full'),
        key=os.path.getmtime,
        reverse=True
    )
    if not list_of_folders:
        logger.error(f"No output folders found in {base_dir}.")
        raise FileNotFoundError(f"No output folders found in {base_dir}.")
    latest_folder = list_of_folders[0]
    logger.debug(f"Latest output folder determined: {latest_folder}")
    return latest_folder


def load_label_encoders(model_folder: Path, categorical_columns: List[str]) -> Dict[str, Any]:
    """Load label encoders for specified categorical columns."""
    encoders = {}
    for col in categorical_columns:
        enc_path = model_folder / f"{col}{ENCODERS_SUFFIX}"
        if not enc_path.exists():
            logger.error(f"Missing encoder for {col}: {enc_path}")
            raise FileNotFoundError(f"Encoder not found: {enc_path}")
        try:
            encoder = load_label_encoder(enc_path)
            encoders[col] = encoder
            logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded encoder for {col} from {enc_path}")
        except Exception as e:
            logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to load encoder for {col}: {e}")
            raise
    return encoders


def build_test_row(comp: Dict[str, Any], out_map: Dict[str, Any], in_map: Dict[str, Any],
                  comp_lookup: Dict[str, Any], top_guids: List[str],
                  top_names: List[str], top_input_params: List[str],
                  use_simple_labels: bool) -> Dict[str, Any]:
    """
    Build a single test data row from a component.
    This function is designed to be used with parallel processing.
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
        if uid in top_guids:
            vec_guids[top_guids.index(uid)] = 1
        if up_name in top_names:
            vec_names[top_names.index(up_name)] = 1

    # One-hot for top input param IDs
    vec_inp = np.zeros(len(top_input_params), dtype=int)
    for p in comp.get("Parameters", []):
        if p.get("ParameterType") == "Input" and p.get("Id"):
            pid = p["Id"]
            if pid in top_input_params:
                vec_inp[top_input_params.index(pid)] = 1

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


def prepare_test_data(filtered_comps: List[Dict[str, Any]], out_map: Dict[str, Any],
                     in_map: Dict[str, Any], comp_lookup: Dict[str, Any],
                     top_guids: List[str], top_names: List[str],
                     top_input_params: List[str], use_simple_labels: bool) -> List[Dict[str, Any]]:
    """Prepare test data rows, potentially using parallel processing."""
    data_rows = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                build_test_row,
                comp,
                out_map,
                in_map,
                comp_lookup,
                top_guids,
                top_names,
                top_input_params,
                use_simple_labels
            )
            for comp in filtered_comps
        ]
        for f in tqdm(futures, desc=f"{Fore.CYAN}Building test rows{Style.RESET_ALL}", unit="comp"):
            result = f.result()
            if result:
                data_rows.append(result)
    return data_rows


def main():
    """Main function to evaluate the ONNX model."""
    args = parse_arguments()

    test_folder = Path(args.test_json_folder)

    if not test_folder.exists():
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Test JSON folder does not exist: {test_folder}")
        sys.exit(1)

    # Determine the latest output folder
    try:
        latest_output_folder = get_latest_output_folder(DEFAULT_OUTPUT_FOLDER)
    except FileNotFoundError as e:
        logger.critical(f"{Fore.RED}[CRITICAL]{Style.RESET_ALL} {e}")
        sys.exit(1)

    # Set paths
    onnx_path = latest_output_folder / MODEL_FILENAME
    config_path = latest_output_folder / FEATURES_CONFIG_FILENAME

    if not onnx_path.exists():
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} ONNX model not found: {onnx_path}")
        sys.exit(1)

    if not config_path.exists():
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Feature config not found: {config_path}")
        sys.exit(1)

    # Load ONNX model
    try:
        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded ONNX model from {onnx_path}")
    except Exception as e:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to load ONNX model: {e}")
        sys.exit(1)

    # Load feature configuration
    try:
        feature_config = load_features_config(config_path)
        logger.debug(f"{Fore.CYAN}[DEBUG]{Style.RESET_ALL} Feature configuration loaded.")
    except Exception as e:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to load feature configuration: {e}")
        sys.exit(1)

    feature_cols = feature_config.get("feature_cols", [])
    top_guids = feature_config.get("top_guids", [])
    top_names = feature_config.get("top_names", [])
    top_input_params = feature_config.get("top_input_params", [])
    use_simple_labels = feature_config.get("use_simple_labels", False)

    # Define categorical columns
    categorical_cols = ["CurrentGUID", "CurrentName", "DownstreamGUIDName"]

    # Load encoders
    try:
        encoders = load_label_encoders(latest_output_folder, categorical_cols)
    except Exception as e:
        logger.critical(f"{Fore.RED}[CRITICAL]{Style.RESET_ALL} {e}")
        sys.exit(1)

    # Load test JSON data
    try:
        comps_test = load_json_files(test_folder, file_limit=EVAL_FILE_LIMIT)
        if not comps_test:
            logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No components found in test JSON folder: {test_folder}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to load test JSON files: {e}")
        sys.exit(1)

    in_map = build_input_map(comps_test)
    out_map = build_output_map(comps_test)
    comp_lookup = build_comp_lookup(comps_test)

    # Filter components with at least one Output parameter
    filtered_comps = [
        c for c in comps_test
        if any(p.get("ParameterType") == "Output" for p in c.get("Parameters", []))
    ]
    if not filtered_comps:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid components after filtering in test data.")
        sys.exit(1)

    logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Number of components after filtering: {len(filtered_comps)}")

    # Prepare test data rows
    try:
        data_rows = prepare_test_data(
            filtered_comps,
            out_map,
            in_map,
            comp_lookup,
            top_guids,
            top_names,
            top_input_params,
            use_simple_labels
        )
    except Exception as e:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to prepare test data: {e}")
        sys.exit(1)

    # Create DataFrame
    df_test = pd.DataFrame(data_rows)
    if df_test.empty:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} No valid test rows generated. Exiting.")
        sys.exit(1)

    logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Test DataFrame shape: {df_test.shape}")

    # Encode categorical features
    for col in categorical_cols:
        df_test[col] = df_test[col].astype(str)
        # Create a mapping from classes to indices
        classes = encoders[col].classes_
        mapping = {cls: idx for idx, cls in enumerate(classes)}
        default_idx = len(encoders[col].classes_)  # Assign unseen labels to this index

        # Log the presence of 'Unknown' in encoder classes
        if "Unknown" in mapping:
            logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} 'Unknown' label found in encoder for '{col}'.")
        else:
            logger.warning(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} 'Unknown' label not found in encoder for '{col}'. Assigning unseen labels to index {default_idx}.")

        # Apply the mapping with a default index for unseen labels
        df_test[col] = df_test[col].apply(lambda x: mapping[x] if x in mapping else default_idx)
        logger.debug(f"{Fore.CYAN}[DEBUG]{Style.RESET_ALL} Encoded column '{col}'. Assigned unseen labels to index {default_idx}.")

    # Drop original string columns and rename encoded columns
    df_test.rename(columns={
        "CurrentGUID": "CurrentGUID_encoded",
        "CurrentName": "CurrentName_encoded",
        "DownstreamGUIDName": "DownstreamGUIDName_encoded"
    }, inplace=True)

    logger.info(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Downstream label distribution in TEST after encoding:")
    print(df_test["DownstreamGUIDName_encoded"].value_counts().head(20))

    # Ensure all feature columns are present
    missing_feats = set(feature_cols) - set(df_test.columns)
    if missing_feats:
        logger.warning(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Missing features in test data: {missing_feats}")
        for mf in missing_feats:
            df_test[mf] = 0  # or appropriate default value
        logger.debug(f"{Fore.CYAN}[DEBUG]{Style.RESET_ALL} Assigned default values to missing features.")

    # Prepare feature matrix and labels
    X_test = df_test[feature_cols].astype(np.float32)
    y_test = df_test["DownstreamGUIDName_encoded"].values

    logger.debug(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    logger.debug(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Number of distinct labels in y_test: {len(np.unique(y_test))}")

    # ONNX Inference
    try:
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: X_test.to_numpy()}
        ort_outs = session.run(None, ort_inputs)
        probs = ort_outs[0]
        logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Inference completed successfully.")
    except Exception as e:
        logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Inference failed: {e}")
        sys.exit(1)

    # Handle single-class output
    if probs.ndim == 1:
        logger.warning(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Single-class output detected. Assigning all predictions to class 0.")
        preds = np.zeros_like(probs, dtype=int)
    else:
        preds = probs.argmax(axis=1)

    # Evaluation Metrics
    acc = accuracy_score(y_test, preds)
    f1_w = f1_score(y_test, preds, average="weighted", zero_division=0)
    logger.info(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} Accuracy: {acc:.4f}")
    logger.info(f"{Fore.GREEN}[RESULT]{Style.RESET_ALL} Weighted F1 Score: {f1_w:.4f}")
    logger.info(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    # Confusion Matrix
    unique_test_labels = np.unique(y_test)
    unique_pred_labels = np.unique(preds)
    if len(unique_test_labels) > 1 and len(unique_pred_labels) > 1:
        cm = confusion_matrix(y_test, preds)
        logger.info(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Confusion Matrix:")
        print(cm)
    else:
        logger.info(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Confusion matrix not displayed (single class in y_test or preds).")

    # Top-5 Predictions for the first sample
    if len(X_test) > 0 and probs.ndim == 2:
        row0 = probs[0]
        top5_idx = row0.argsort()[::-1][:5]
        top5_scores = row0[top5_idx]
        actual_enc = y_test[0]
        try:
            actual_lbl = encoders["DownstreamGUIDName"].inverse_transform([actual_enc])[0]
        except Exception as e:
            logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to inverse transform label '{actual_enc}': {e}")
            actual_lbl = "Unknown"

        logger.info(f"\n{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Sample #0 Actual DownstreamGUIDName: {actual_lbl}")
        for rank, (idx_c, sc) in enumerate(zip(top5_idx, top5_scores), start=1):
            try:
                lbl_str = encoders["DownstreamGUIDName"].inverse_transform([idx_c])[0]
            except Exception as e:
                logger.error(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Failed to inverse transform index '{idx_c}': {e}")
                lbl_str = "Unknown"
            print(f"  Rank {rank}: {lbl_str}, score={sc:.4f}")
    else:
        logger.info(f"{Fore.MAGENTA}[DEBUG]{Style.RESET_ALL} Top-5 predictions not available (single class or empty X_test).")


if __name__ == "__main__":
    main()
