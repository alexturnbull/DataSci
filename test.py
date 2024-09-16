
from datetime import datetime
import subprocess
import os

def get_kaggle_data():
    download_path = '/home/alex/Projects/DATALAKE/BRONZE/KAGGLE_SALES_DATA'
    zip_file = os.path.join(download_path, 'store-sales-time-series-forecasting.zip')
    
    try:
        # Downloading the dataset
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 'store-sales-time-series-forecasting', '-p', download_path], check=True)
        
        # Unzipping the downloaded file
        subprocess.run(['unzip', '-o', zip_file, '-d', download_path], check=True)
        
        # Optional: Remove the zip file after extraction
        os.remove(zip_file)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        raise
  
get_kaggle_data()