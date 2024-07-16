import modin.pandas as pd
import kaggle
import zipfile
import os

# Dwonload and unzip csv file

# Authenticate using the kaggle.json file
kaggle.api.authenticate()

# Define paths
competition_name = 'store-sales-time-series-forecasting'
download_path = 'Datasets'
zip_file_path = os.path.join(download_path, f'{competition_name}.zip')

# Download the competition dataset
try:
    kaggle.api.competition_download_files(competition_name, path=download_path)
    print("Dataset downloaded successfully")
except Exception as e:
    print(f"Error downloading dataset: {e}")

# Unzip the downloaded file
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    print("Extraction completed successfully")
except Exception as e:
    print(f"Error extracting dataset: {e}")


# write to parquet 


# Define the paths
csv_file_path = 'path/to/your/file.csv'
parquet_file_path = 'path/to/your/file.parquet'

# Read the CSV file into a DataFrame
new_data = pd.read_csv(csv_file_path)

# Check if the Parquet file already exists
if os.path.exists(parquet_file_path):
    # Read the existing Parquet file
    existing_data = pd.read_parquet(parquet_file_path)
    
    # Append the new data to the existing data
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
else:
    # If the Parquet file doesn't exist, the new data is the combined data
    combined_data = new_data

# Write the combined DataFrame to the Parquet file
combined_data.to_parquet(parquet_file_path, engine='pyarrow', index=False)

print(f"Data has been successfully appended to the Parquet file at '{parquet_file_path}'")
