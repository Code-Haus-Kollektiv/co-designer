"""
xgboost_utilities.py

Shared utility functions for reading JSON chunk files,
building forward/backward maps, extracting numeric features, etc.
No circular imports or references here.
"""

import os
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from sklearn.preprocessing import LabelEncoder

try:
    import colorama
    from colorama import Fore, Style
    colorama.init(strip=False, convert=True)
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

def read_single_file(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
        return data.get("Components", [])

def load_json_files(folder_path, file_limit=None):
    """Load JSON files in parallel, extracting all 'Components' from each file."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if file_limit is not None:
        files = files[:file_limit]

    file_paths = [os.path.join(folder_path, fn) for fn in files]
    components = []
    print(f"[INFO] Loading JSON from: {folder_path}")
    with ThreadPoolExecutor() as executor:
        for comps in tqdm(
            executor.map(read_single_file, file_paths),
            total=len(file_paths),
            desc=f"{Fore.CYAN}Reading JSON files{Style.RESET_ALL}",
            unit="file"
        ):
            components.extend(comps)

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Total components loaded:", len(components))
    return components

def build_input_map(components):
    """For each Input Param ID => list of component IDs that consume it."""
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
    """For each Output Param ID => list of component IDs that produce it."""
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
    """comp_id -> entire component object."""
    return {c["Id"]: c for c in components if c.get("Id")}

def get_upstream_ids(current_comp, output_map):
    """Return all component IDs that produce the inputs used by current_comp."""
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
    """Return all component IDs that consume the outputs produced by current_comp."""
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
    """Return basic numeric features: Name, #Params, #Input, #Output."""
    params = comp.get("Parameters", [])
    inp = [p for p in params if p.get("ParameterType") == "Input"]
    return {
        "Name": comp.get("Name", "Unknown"),
        "NumParams": len(params),
        "NumInput": len(inp),
        "NumOutput": len(params) - len(inp)
    }

def create_dated_output_subfolder(base_folder, model_name):
    """Create a new subfolder with a timestamp + index in base_folder."""
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
    """Save a LabelEncoder's classes to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(encoder.classes_.tolist(), f, indent=2)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved LabelEncoder to {filepath}")

def load_label_encoder(filepath):
    """Load a LabelEncoder's classes from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        classes_list = json.load(f)
    le = LabelEncoder()
    import numpy as np
    le.classes_ = np.array(classes_list, dtype=object)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded LabelEncoder from {filepath}")
    return le

def save_features_config(config, filepath):
    """Save feature configuration to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Saved feature configuration to {filepath}")

def load_features_config(filepath):
    """Load feature configuration from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Loaded feature configuration from {filepath}")
    return config
