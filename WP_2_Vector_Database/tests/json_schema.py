import json
from genson import SchemaBuilder

builder = SchemaBuilder()

# Load the input JSON data
with open(r'WP_2_Vector_Database\json_chunks\1rHcXSS6zdJ90VFWoH44vwqnMCO.json') as f:
    data = json.load(f)

# Add the loaded data to the schema builder
builder.add_object(data)

# Generate the schema
schema = builder.to_schema()

# Save the schema to a file
with open(r'WP_2_Vector_Database\output\output_schema.json', 'w') as schema_file:
    json.dump(schema, schema_file, indent=2)

print("Schema has been saved to 'output_schema.json'")
