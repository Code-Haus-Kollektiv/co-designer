"""
xgboost_utilities.py

Shared utility functions for reading JSON chunk files,
building forward/backward maps, extracting numeric features, etc.
No circular imports.
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
from sklearn.preprocessing import LabelEncoder

from colorama import Fore, Style, init
init(autoreset=True)

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = ColoredFormatter("%(asctime)s [%(levelname)s] %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc="", unit=""):
        return iterable

def read_single_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("Components", [])
    except Exception as e:
        logger.error("Error reading file %s: %s", path, e)
        return []

def load_json_files(folder_path, file_limit=None):
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if file_limit is not None:
        files = files[:file_limit]
    file_paths = [os.path.join(folder_path, fn) for fn in files]
    components = []
    logger.info("Loading JSON from: %s", folder_path)
    with ThreadPoolExecutor() as executor:
        for comps in tqdm(executor.map(read_single_file, file_paths),
                          total=len(file_paths),
                          desc="Reading JSON files",
                          unit="file"):
            components.extend(comps)
    logger.info("Total components loaded: %d", len(components))
    return components

def build_input_map(components):
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
    return {c["Id"]: c for c in components if c.get("Id")}

def get_upstream_ids(current_comp, output_map):
    cid = current_comp.get("Id")
    if not cid:
        return []
    input_param_ids = [p["Id"] for p in current_comp.get("Parameters", [])
                       if p.get("ParameterType") == "Input" and p.get("Id")]
    ups = set()
    for pid in input_param_ids:
        if pid in output_map:
            ups.update(output_map[pid])
    return list(ups)

def get_downstream_ids(current_comp, input_map):
    cid = current_comp.get("Id")
    if not cid:
        return []
    output_param_ids = [p["Id"] for p in current_comp.get("Parameters", [])
                        if p.get("ParameterType") == "Output" and p.get("Id")]
    downs = set()
    for pid in output_param_ids:
        if pid in input_map:
            downs.update(input_map[pid])
    return list(downs)

def extract_numeric_features(comp):
    params = comp.get("Parameters", [])
    inp = [p for p in params if p.get("ParameterType") == "Input"]
    return {
        "Name": comp.get("Name", "Unknown"),
        "NumParams": len(params),
        "NumInput": len(inp),
        "NumOutput": len(params) - len(inp)
    }

def create_dated_output_subfolder(base_folder, model_name):
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    idx = 1
    while True:
        subfolder_name = f"{date_str}_{idx}_{model_name}"
        subfolder_path = os.path.join(base_folder, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            return subfolder_path
        idx += 1

def save_label_encoder(encoder, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)
    logger.info("Saved LabelEncoder to %s", filepath)

def load_label_encoder(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        classes_list = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes_list, dtype=object)
    logger.info("Loaded LabelEncoder from %s", filepath)
    return le

def save_features_config(config, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Saved feature configuration to %s", filepath)

def load_features_config(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Loaded feature configuration from %s", filepath)
    return config
