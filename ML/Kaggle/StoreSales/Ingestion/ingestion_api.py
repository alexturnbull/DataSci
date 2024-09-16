import modin.pandas as pd
import kaggle
import zipfile
import os




def get_kaggle_data():
    kaggle.api.authenticate()
    competition_name = 'store-sales-time-series-forecasting'
    download_path = '/home/alex/Projects/DataSci/ML/Kaggle/StoreSales/Datasets'
    zip_file_path = os.path.join(download_path, f'{competition_name}.zip')
    try:
        kaggle.api.competition_download_files(competition_name, path=download_path)
        print("Dataset downloaded successfully")
    except Exception as e:
            print(f"Error downloading dataset: {e}")

    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
            print("Extraction completed successfully")
    except Exception as e:
        print(f"Error extracting dataset: {e}")

