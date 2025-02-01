import os
import logging
import sys
import time
from typing import List, Dict, Any

import onnx
import onnxruntime as ort
import numpy as np
import pandas as pd
from colorama import Fore, Style, init
import json
from pathlib import Path
import glob

# -----------------------------
# Configuration Parameters
# -----------------------------

# Base output directory
OUTPUT_BASE_DIR = Path(r"C:\Users\Lasat\Documents\GitHub\co-designer\WP_2_Vector_Database\output")

# Flag to determine whether to use GPU for inference
USE_GPU = True  # Set to False to use CPU

# Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = 'DEBUG'

# -----------------------------
# End of Configuration
# -----------------------------


# Initialize colorama
init(autoreset=True)


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


class ONNXModelTester:
    def __init__(self, output_base_dir: Path, use_gpu: bool = False):
        self.output_base_dir = output_base_dir
        self.use_gpu = use_gpu
        self.session = None
        self.input_details = []
        self.output_details = []
        self.feature_config = {}
        self.index_to_label = {}
        self.encoders = {}
        self.latest_output_folder = None
        self.model_path = None
        self.features_config_path = None
        self.index_to_label_path = None
        self.guid_encoder_path = None
        self.name_encoder_path = None

    def get_latest_output_folder(self) -> Path:
        """
        Retrieves the latest output folder based on modification time.
        Assumes that output folders end with '_Full'.
        """
        list_of_folders = sorted(
            self.output_base_dir.glob('*_Full'),
            key=os.path.getmtime,
            reverse=True
        )
        if not list_of_folders:
            logger.error(f"No output folders found in {self.output_base_dir}.")
            raise FileNotFoundError(f"No output folders found in {self.output_base_dir}.")
        latest_folder = list_of_folders[0]
        logger.debug(f"Latest output folder determined: {latest_folder}")
        return latest_folder

    def update_paths_dynamically(self):
        """
        Updates the model and configuration paths based on the latest output folder.
        """
        self.latest_output_folder = self.get_latest_output_folder()
        self.model_path = self.latest_output_folder / "xgboost_model.onnx"
        self.features_config_path = self.latest_output_folder / "features_config.json"
        self.index_to_label_path = self.latest_output_folder / "index_to_label.json"
        self.guid_encoder_path = self.latest_output_folder / "CurrentGUID_encoder.json"
        self.name_encoder_path = self.latest_output_folder / "CurrentName_encoder.json"
        logger.info(f"Paths updated to latest output folder: {self.latest_output_folder}")

    def verify_model_path(self):
        if not self.model_path.is_file():
            logger.error(f"Model file does not exist: {self.model_path}")
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        logger.debug(f"Model path verified: {self.model_path}")

    def load_and_check_model(self):
        logger.info("Loading ONNX model...")
        try:
            model = onnx.load(str(self.model_path))
            logger.debug("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load the ONNX model: {e}")
            raise

        logger.info("Checking model integrity...")
        try:
            onnx.checker.check_model(model)
            logger.info("ONNX model is structurally valid.")
        except onnx.checker.ValidationError as e:
            logger.error(f"Model validation failed: {e}")
            raise

    def setup_execution_providers(self) -> List[str]:
        logger.info("Setting up execution providers...")
        available_providers = ort.get_available_providers()
        logger.debug(f"Available ONNX Runtime providers: {available_providers}")

        if self.use_gpu and 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider']
            logger.info("CUDAExecutionProvider selected for GPU acceleration.")
        else:
            providers = ['CPUExecutionProvider']
            if self.use_gpu:
                logger.warning("CUDAExecutionProvider not available. Falling back to CPUExecutionProvider.")
            else:
                logger.info("CPUExecutionProvider selected.")

        return providers

    def create_inference_session(self, providers: List[str]):
        logger.info("Creating ONNX Runtime inference session...")
        try:
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            device = 'GPU' if 'CUDAExecutionProvider' in providers else 'CPU'
            logger.info(f"ONNX model successfully loaded on {device}.")
        except Exception as e:
            logger.error(f"Failed to create InferenceSession: {e}")
            raise

        # Retrieve input and output details
        self.input_details = self.session.get_inputs()
        self.output_details = self.session.get_outputs()
        logger.debug(f"Model Inputs: {[input.name for input in self.input_details]}")
        logger.debug(f"Model Outputs: {[output.name for output in self.output_details]}")

    def load_feature_config(self):
        logger.info(f"Loading feature configuration from: {self.features_config_path}")
        if not self.features_config_path.is_file():
            logger.error(f"Feature configuration file does not exist: {self.features_config_path}")
            raise FileNotFoundError(f"Feature configuration file not found: {self.features_config_path}")
        with self.features_config_path.open('r', encoding='utf-8') as f:
            self.feature_config = json.load(f)
        logger.debug(f"Feature configuration loaded: {self.feature_config}")

    def load_index_to_label_map(self):
        logger.info(f"Loading index-to-label mapping from: {self.index_to_label_path}")
        if not self.index_to_label_path.is_file():
            logger.error(f"Index-to-label mapping file does not exist: {self.index_to_label_path}")
            raise FileNotFoundError(f"Index-to-label mapping file not found: {self.index_to_label_path}")
        with self.index_to_label_path.open('r', encoding='utf-8') as f:
            self.index_to_label = json.load(f)
        logger.debug(f"Index-to-label mapping loaded: {self.index_to_label}")

    def load_encoders(self):
        logger.info("Loading encoders...")
        encoder_paths = {
            "CurrentGUID": self.guid_encoder_path,
            "CurrentName": self.name_encoder_path
        }
        for key, path in encoder_paths.items():
            if not path.is_file():
                logger.error(f"Encoder file does not exist: {path}")
                raise FileNotFoundError(f"Encoder file not found: {path}")
            with path.open('r', encoding='utf-8') as f:
                encoder_data = json.load(f)
                if isinstance(encoder_data, dict) and "classes_" in encoder_data:
                    classes = encoder_data.get("classes_", [])
                elif isinstance(encoder_data, list):
                    classes = encoder_data
                else:
                    logger.error(f"Unsupported encoder format in file: {path}")
                    raise ValueError(f"Unsupported encoder format in file: {path}")
                if not classes:
                    logger.error(f"No classes found for encoder '{key}' in file: {path}")
                    raise ValueError(f"No classes found for encoder '{key}' in file: {path}")
                self.encoders[key] = classes
                logger.debug(f"Encoder '{key}' loaded with classes: {classes[:5]}...")  # Show first 5 classes
        logger.info("All encoders loaded successfully.")

    def prepare_input_data(self, input_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Prepare input data based on feature configuration and encoders.
        """
        logger.info("Preparing input data based on feature configuration and encoders...")
        input_features = self.feature_config.get("feature_cols", [])
        prepared_inputs = {}

        for idx, feature in enumerate(input_features):
            # Map feature names to model input names (assuming 'input' is a single tensor)
            # Since model input is a single tensor, we need to prepare a single numpy array with all features
            # This implementation assumes that the model expects a single input tensor with all features
            # Therefore, we'll collect all feature values into a single list
            pass  # We'll handle this differently below

        # Since the model input is a single tensor named 'input', we'll prepare a single array
        input_values = []
        for feature in input_features:
            value = input_dict.get(feature, 0)  # Default to 0 if feature is missing

            # Handle encoded features
            if feature == "CurrentGUID_encoded":
                guid = input_dict.get("CurrentGUID", "Unknown_GUID")
                if guid in self.encoders["CurrentGUID"]:
                    encoded_value = self.encoders["CurrentGUID"].index(guid)
                else:
                    logger.warning(f"GUID '{guid}' not found in encoder. Assigning -1.")
                    encoded_value = -1  # Handle unseen GUIDs
                value = encoded_value

            elif feature == "CurrentName_encoded":
                name = input_dict.get("CurrentName", "Unknown_Name")
                if name in self.encoders["CurrentName"]:
                    encoded_value = self.encoders["CurrentName"].index(name)
                else:
                    logger.warning(f"Name '{name}' not found in encoder. Assigning -1.")
                    encoded_value = -1  # Handle unseen Names
                value = encoded_value

            # Append the value to the input list
            input_values.append(value)
            logger.debug(f"Feature '{feature}' processed with value: {value}")

        # Convert the list to a numpy array with shape (1, num_features)
        prepared_input_array = np.array([input_values], dtype=np.float32)
        prepared_inputs[self.input_details[0].name] = prepared_input_array

        logger.info("Input data prepared successfully.")
        logger.debug(f"Prepared input array shape: {prepared_input_array.shape}")
        logger.debug(f"Prepared input array: {prepared_input_array}")

        return prepared_inputs

    def prepare_dummy_inputs(self) -> Dict[str, Any]:
        """
        Prepare dummy inputs for inference based on feature configuration.
        """
        logger.info("Preparing dummy inputs for inference...")
        dummy_input = {}
        feature_cols = self.feature_config.get("feature_cols", [])

        for feature in feature_cols:
            # Assign default or random values based on feature type
            if "encoded" in feature:
                if "CurrentGUID_encoded" in feature:
                    dummy_input["CurrentGUID"] = "Unknown_GUID"
                elif "CurrentName_encoded" in feature:
                    dummy_input["CurrentName"] = "Unknown_Name"
            elif feature.startswith("UpGUID") or feature.startswith("UpName") or feature.startswith("InpParam"):
                dummy_input[feature] = 0  # Binary features
            else:
                dummy_input[feature] = 0  # Numerical features

        logger.debug(f"Dummy input data: {dummy_input}")
        return dummy_input

    def run_inference(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        if self.session is None:
            logger.error("Inference session is not initialized.")
            raise RuntimeError("Inference session not initialized.")

        logger.info("Running inference...")
        start_time = time.time()
        try:
            outputs = self.session.run(None, inputs)
            inference_time = time.time() - start_time
            logger.info(f"Inference completed in {inference_time:.4f} seconds.")
            return outputs
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

    def display_output(self, outputs: List[np.ndarray]):
        for idx, output in enumerate(outputs):
            output_name = self.output_details[idx].name
            logger.info(f"Output '{output_name}' shape: {output.shape}")
            logger.debug(f"Output '{output_name}' values:\n{output}")

    def decode_predictions(self, outputs: List[np.ndarray]) -> List[str]:
        """
        Decode the model's output probabilities to the corresponding labels.
        Assumes that the second output is the probabilities for each class.
        """
        if not outputs:
            logger.error("No outputs to decode.")
            return []

        if len(outputs) < 2:
            logger.error("Expected at least two outputs ('label' and 'probabilities').")
            return []

        probabilities = outputs[1]  # Assuming 'probabilities' is the second output
        logger.debug(f"Raw output probabilities: {probabilities}")

        # Get the index of the highest probability for each sample
        predicted_indices = np.argmax(probabilities, axis=1)
        logger.debug(f"Predicted class indices: {predicted_indices}")

        # Map indices to labels
        predicted_labels = [self.index_to_label.get(str(idx), "Unknown") for idx in predicted_indices]
        logger.info(f"Predicted labels: {predicted_labels}")
        return predicted_labels

    def test_model_inference(self, input_data: Dict[str, Any] = None):
        self.verify_model_path()
        self.load_and_check_model()
        providers = self.setup_execution_providers()
        self.create_inference_session(providers)
        self.load_feature_config()
        self.load_index_to_label_map()
        self.load_encoders()

        if input_data is None:
            logger.info("No input data provided. Using dummy inputs for inference.")
            input_data = self.prepare_dummy_inputs()

        # Prepare inputs based on feature configuration and encoders
        prepared_inputs = self.prepare_input_data(input_data)

        # Run inference
        outputs = self.run_inference(prepared_inputs)

        # Display output shapes and values
        self.display_output(outputs)

        # Decode predictions
        predicted_labels = self.decode_predictions(outputs)
        logger.info(f"Decoded Predicted Labels: {predicted_labels}")

    def test_model(self, base_dir: Path):
        self.update_paths_dynamically()
        self.test_model_inference()


def main():
    tester = ONNXModelTester(
        output_base_dir=OUTPUT_BASE_DIR,
        use_gpu=USE_GPU
    )
    try:
        tester.test_model(base_dir=OUTPUT_BASE_DIR)
        logger.info("ONNX model test completed successfully.")
    except Exception as e:
        logger.critical(f"ONNX model test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
