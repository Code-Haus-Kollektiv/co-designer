import xmltodict
import json
import os
from typing import Dict, Any

def extract_object_chunks(xml_file: str, output_folder: str) -> None:
    """
    Extract and save chunks with name="Object" from the XML file into individual JSON files.

    :param xml_file: Path to the input XML file.
    :param output_folder: Directory where JSON files will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Read and parse the XML file
    with open(xml_file, 'r', encoding='utf-8') as file:
        xml_content = file.read()
        data_dict = xmltodict.parse(xml_content)

    def save_json_file(data: Dict, index: int) -> None:
        """
        Save a dictionary as a JSON file.

        :param data: The data to save.
        :param index: Index for the JSON file name.
        """
        json_filename = os.path.join(output_folder, f"object_{index}.json")
        with open(json_filename, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Saved: {json_filename}")

    def find_object_chunks(node: Any, index: int = 1) -> int:
        """
        Recursively search for and save chunks named "Object".

        :param node: The current XML node (dictionary or list).
        :param index: Current index for naming JSON files.
        :return: Updated index after processing.
        """
        if isinstance(node, dict):
            for key, value in node.items():
                # Check if this is a chunk with name="Object"
                if key == "chunk" and isinstance(value, dict) and value.get("@name") == "Object":
                    save_json_file(value, index)
                    index += 1
                elif key == "chunk" and isinstance(value, list):
                    # Handle list of chunks
                    for chunk in value:
                        if chunk.get("@name") == "Object":
                            save_json_file(chunk, index)
                            index += 1
                # Recursively search deeper
                index = find_object_chunks(value, index)
        elif isinstance(node, list):
            for item in node:
                index = find_object_chunks(item, index)
        return index

    # Start processing from the root node
    print("Starting processing...")
    find_object_chunks(data_dict)
    print("Processing completed.")

# Example usage
if __name__ == "__main__":
    xml_file_path = 'test_data.xml'  # Path to your XML file
    output_directory = 'json_chunks'  # Directory to save JSON files

    extract_object_chunks(xml_file_path, output_directory)
