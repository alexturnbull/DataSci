import pandas as pd
import os
import json
import yaml

def read_config(config_file):
    """Read the configuration file."""
    with open(config_file, 'r') as file:
        if config_file.endswith('.json'):
            return json.load(file)
        elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return yaml.safe_load(file)
        else:
            raise ValueError("Unsupported config file format")

def process_files(config):
    """Process each file as per the configuration."""
    for item in config.get('files', []):
        csv_file_path = item.get('csv')
        parquet_file_path = item.get('parquet')
        mode = item.get('mode', 'append')  # Default to 'append' if no mode is specified

        if not csv_file_path or not parquet_file_path:
            print(f"Invalid configuration: {item}")
            continue

        # Read the new CSV data
        try:
            new_data = pd.read_csv(csv_file_path)
            print(f"Read CSV file: {csv_file_path}")
        except Exception as e:
            print(f"Error reading CSV file {csv_file_path}: {e}")
            continue

        if mode == 'append':
            if os.path.exists(parquet_file_path):
                # Read existing data and append new data
                try:
                    existing_data = pd.read_parquet(parquet_file_path)
                    print(f"Read existing Parquet file: {parquet_file_path}")
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                except Exception as e:
                    print(f"Error reading Parquet file {parquet_file_path}: {e}")
                    combined_data = new_data
            else:
                combined_data = new_data
        elif mode == 'refresh':
            combined_data = new_data
        else:
            print(f"Invalid mode: {mode}. Use 'append' or 'refresh'.")
            continue

        # Write the combined DataFrame to the Parquet file
        try:
            combined_data.to_parquet(parquet_file_path, engine='pyarrow', index=False)
            print(f"Updated Parquet file: {parquet_file_path}")
        except Exception as e:
            print(f"Error writing Parquet file {parquet_file_path}: {e}")

if __name__ == "__main__":
    # Hard-coded path to your configuration file
    config_file_path = '/home/alex/Projects/DataSci/ML/Kaggle/StoreSales/Ingestion/config.yml'  # Change this to your actual config file path

    # Read the configuration
    config = read_config(config_file_path)

    # Process the files according to the configuration
    process_files(config)
