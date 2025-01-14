import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score

# For colored prints
try:
    import colorama
    from colorama import Fore, Style
    colorama.init(autoreset=True)  # Initializes colorama (especially important on Windows)
except ImportError:
    print("colorama is not installed. Please install it via 'pip install colorama' for colored output.")
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
    class Style:
        RESET_ALL = ""

# For progress bars
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm is not installed. Please install it via 'pip install tqdm' to see progress bars.")
    # We can define a mock tqdm in that case:
    def tqdm(iterable, desc="", unit=""):
        return iterable

# ------------------------------------------------------------------------------
# 1) Read all JSON files from the folder using a progress bar
# ------------------------------------------------------------------------------
JSON_FOLDER = r".\WP_2_Vector_Database\json_chunks\Results"
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} JSON_FOLDER path: {JSON_FOLDER}")

all_components = []
json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

# Use tqdm to show progress of reading files
for filename in tqdm(json_files, desc="Reading JSON files", unit="file"):
    file_path = os.path.join(JSON_FOLDER, filename)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # assume there's a "Components" key
        comps = data.get("Components", [])
        all_components.extend(comps)

print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Total JSON files processed: {len(json_files)}")
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Parsed total components: {len(all_components)}\n")

# ------------------------------------------------------------------------------
# 2) Derive "NextComponent" from output parameters, with a progress bar
# ------------------------------------------------------------------------------
def find_downstream_components(current_component, all_comps):
    downstream_ids = set()
    output_params = [
        p for p in current_component.get("Parameters", [])
        if p.get("ParameterType") == "Output"
    ]
    
    for outp in output_params:
        connected_ids = outp.get("ConnectedIds", []) or []
        for cid in connected_ids:
            for comp in all_comps:
                for param in comp.get("Parameters", []):
                    if param.get("Id") == cid and param.get("ParameterType") == "Input":
                        downstream_ids.add(comp.get("Id", ""))
                        break
    return list(downstream_ids)

def derive_next_component_name(current_component, all_comps):
    downstream_ids = find_downstream_components(current_component, all_comps)
    if not downstream_ids:
        return None
    first_downstream_id = downstream_ids[0]
    
    for comp in all_comps:
        if comp.get("Id") == first_downstream_id:
            return comp.get("Name", None)
    return None

# Use tqdm to show progress of attaching NextComponent
for comp in tqdm(all_components, desc="Deriving next components", unit="comp"):
    comp["NextComponent"] = derive_next_component_name(comp, all_components)

# ------------------------------------------------------------------------------
# 3) Feature Extraction
# ------------------------------------------------------------------------------
def extract_features_from_component(comp):
    name = comp.get("Name", "")
    category = comp.get("Category", "")
    subcategory = comp.get("SubCategory", "")
    
    parameters = comp.get("Parameters", [])
    num_params = len(parameters)
    num_input = sum(1 for p in parameters if p.get("ParameterType") == "Input")
    num_output = sum(1 for p in parameters if p.get("ParameterType") == "Output")
    total_connections = sum(p.get("ConnectedCount", 0) for p in parameters)
    
    next_comp = comp.get("NextComponent", None)  # derived from outputs

    return {
        "Id": comp.get("Id", ""),
        "Name": name,
        "Category": category,
        "SubCategory": subcategory,
        "NumParams": num_params,
        "NumInput": num_input,
        "NumOutput": num_output,
        "TotalConnections": total_connections,
        "NextComponent": next_comp
    }

rows = [extract_features_from_component(c) for c in all_components]
df = pd.DataFrame(rows)

print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} DataFrame shape: {df.shape}")
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} First 5 rows of DataFrame:\n{df.head()}\n")

# Optional: see how many unique NextComponents we have
unique_next = df["NextComponent"].nunique(dropna=True)
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Unique NextComponents (excluding NaN): {unique_next}")

# ------------------------------------------------------------------------------
# 4) Encode Categorical Features
# ------------------------------------------------------------------------------
name_enc = LabelEncoder()
cat_enc = LabelEncoder()
subcat_enc = LabelEncoder()
next_enc = LabelEncoder()

df["Name_encoded"] = name_enc.fit_transform(df["Name"].astype(str))
df["Category_encoded"] = cat_enc.fit_transform(df["Category"].astype(str))
df["SubCategory_encoded"] = subcat_enc.fit_transform(df["SubCategory"].astype(str))

# Notice some NextComponent might be None (or NaN if cast to string). Replace with a placeholder.
df["NextComponent"] = df["NextComponent"].fillna("NoDownstream")
df["NextComponent_encoded"] = next_enc.fit_transform(df["NextComponent"].astype(str))

print(f"{Fore.GREEN}[DEBUG]{Style.RESET_ALL} Distribution of NextComponent:")
print(df["NextComponent"].value_counts(), "\n")

X_cols = [
    "Name_encoded", 
    "Category_encoded", 
    "SubCategory_encoded", 
    "NumParams",
    "NumInput", 
    "NumOutput",
    "TotalConnections"
]
X = df[X_cols]
y = df["NextComponent_encoded"]

# ------------------------------------------------------------------------------
# 5) Split Train/Test
# ------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"{Fore.GREEN}[DEBUG]{Style.RESET_ALL} Training set size: {len(X_train)}")
print(f"{Fore.GREEN}[DEBUG]{Style.RESET_ALL} Test set size: {len(X_test)}\n")

# ------------------------------------------------------------------------------
# 6) XGBoost Training
# ------------------------------------------------------------------------------
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

num_classes = df["NextComponent_encoded"].nunique()

params = {
    "objective": "multi:softmax",
    "eval_metric": "mlogloss",
    "num_class": num_classes,
    "seed": 42
}

print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Starting XGBoost training...")
bst = xgb.train(params, dtrain, num_boost_round=50)
print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} XGBoost training completed.\n")

# ------------------------------------------------------------------------------
# 7) Evaluate
# ------------------------------------------------------------------------------
preds = bst.predict(dtest)
acc = accuracy_score(y_test, preds)
print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} Test Accuracy: {acc:.4f}\n")

# ------------------------------------------------------------------------------
# 8) Predict on a new component
# ------------------------------------------------------------------------------
new_comp = {
    "Id": "test1234",
    "Name": "Line SDL",
    "Category": "Curve",
    "SubCategory": "Primitive",
    "Parameters": [
        {"ParameterType": "Input", "ConnectedCount": 1},
        {"ParameterType": "Output", "ConnectedCount": 2}
    ]
}

def featurize_single(comp):
    feat = extract_features_from_component(comp)
    return {
        "Name_encoded": name_enc.transform([feat["Name"]])[0],
        "Category_encoded": cat_enc.transform([feat["Category"]])[0],
        "SubCategory_encoded": subcat_enc.transform([feat["SubCategory"]])[0],
        "NumParams": feat["NumParams"],
        "NumInput": feat["NumInput"],
        "NumOutput": feat["NumOutput"],
        "TotalConnections": feat["TotalConnections"]
    }

feat_dict = featurize_single(new_comp)
X_new = pd.DataFrame([feat_dict])
dnew = xgb.DMatrix(X_new)

pred_class = int(bst.predict(dnew)[0])
pred_label = next_enc.inverse_transform([pred_class])[0]

print(f"{Fore.MAGENTA}[PREDICTION]{Style.RESET_ALL} Predicted next component name: {pred_label}")

# ------------------------------------------------------------------------------
# 9) Convert XGBoost model to ONNX using hummingbird-ml
# ------------------------------------------------------------------------------
try:
    from hummingbird.ml import convert
    from hummingbird.ml import constants
    import onnx

    print(f"{Fore.CYAN}[INFO]{Style.RESET_ALL} Converting XGBoost model to ONNX...")

    # Convert the booster to ONNX
    # Note: If X_train is a DataFrame, pass X_train.values to ensure we have a numpy array.
    model_hb_onnx = convert(
        bst,       # XGBoost booster
        "onnx",    # desired backend
        X_train.values,  # sample input for shape inference
        extra_config={constants.FILL_NA: True}
    )

    # Save the ONNX model
    onnx.save(model_hb_onnx.model, "xgboost_model.onnx")
    print(f"{Fore.YELLOW}[RESULT]{Style.RESET_ALL} ONNX model saved as: xgboost_model.onnx")

except ImportError as e:
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Could not import hummingbird or onnx. "
          f"Please install with:\n   pip install hummingbird-ml onnx onnxruntime\nError detail: {e}")
except Exception as e:
    print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} Unexpected error during ONNX conversion: {e}")

# ------------------------------------------------------------------------------
# Optional: Test the ONNX model using onnxruntime
# ------------------------------------------------------------------------------
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("xgboost_model.onnx")
input_name = session.get_inputs()[0].name

# Suppose we want to do the same single prediction we did with X_new
pred_onnx = session.run(None, {input_name: X_new.values.astype(np.float32)})
print("ONNX model prediction:", pred_onnx[0])
