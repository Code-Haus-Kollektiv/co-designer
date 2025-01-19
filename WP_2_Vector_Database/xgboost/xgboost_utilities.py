import os
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime

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

def load_json_files(folder_path, file_limit=None):
    """
    Load JSON files from 'folder_path', extracting all 'Components' objects.
    If file_limit is not None, only load up to that many files.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if file_limit is not None:
        files = files[:file_limit]

    components = []
    for filename in tqdm(files, desc=f"{Fore.CYAN}Reading JSON files{Style.RESET_ALL}", unit="file"):
        path = os.path.join(folder_path, filename)
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
            comps = data.get("Components", [])
            components.extend(comps)

    print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} Total components loaded: {len(components)}")
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
    params = comp.get("Parameters", [])
    input_params = [p for p in params if p.get("ParameterType") == "Input"]
    return {
        "Name": comp.get("Name", "Unknown"),
        "NumParams": len(params),
        "NumInput": len(input_params),
        "NumOutput": len(params) - len(input_params)
    }

def create_dated_output_subfolder(base_folder, model_name):
    date_str = datetime.now().strftime("%Y%m%d")
    index = 1
    while True:
        subfolder_name = f"{date_str}_{index}_{model_name}"
        subfolder_path = os.path.join(base_folder, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)
            return subfolder_path
        index += 1
