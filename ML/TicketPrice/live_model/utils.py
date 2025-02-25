import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def convert_to_minutes(duration):
    # Extract hours and minutes using regex
    hours = re.search(r'(\d+)h', duration)
    minutes = re.search(r'(\d+)m', duration)
    
    # Convert to integer values, defaulting to 0 if missing
    total_minutes = (int(hours.group(1)) * 60 if hours else 0) + (int(minutes.group(1)) if minutes else 0)
    
    return total_minutes

def clean_and_convert_time(time_str):
    if isinstance(time_str, str) and ' ' in time_str:  # Check if it's a string and contains a space
        return None  # Or assign a default time like '00:00'
    try:
        return pd.to_datetime(time_str, format='%H:%M', errors='coerce')
    except Exception:
        return None  # If conversion fails, return None

def preprocess_date_and_time(data):
    # Preprocessing Date and Time columns
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], format='%d/%m/%Y')
    data['Journey_day'] = data['Date_of_Journey'].dt.day
    data['Journey_month'] = data['Date_of_Journey'].dt.month

    data['Dep_Time'] = data['Dep_Time'].apply(clean_and_convert_time)
    data['Dep_hour'] = data['Dep_Time'].dt.hour
    data['Dep_minute'] = data['Dep_Time'].dt.minute

    data['Arrival_Time'] = data['Arrival_Time'].apply(clean_and_convert_time)
    data['Arrival_hour'] = data['Arrival_Time'].dt.hour
    data['Arrival_minute'] = data['Arrival_Time'].dt.minute

    return data

def encode_categorical_features(data, categorical_columns):
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le  # Save label encoder for possible inverse transformation later
    return data, label_encoders

def normalize_numerical_columns(data, numerical_columns, scaler=None):
    if not scaler:
        scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data, scaler

def map_total_stops(data):
    stops_mapping = {
        'non-stop': 0,
        '1 stop': 1,
        '2 stops': 2,
        '3 stops': 3,
        '4 stops': 4
    }
    data['Total_Stops'] = data['Total_Stops'].map(stops_mapping)
    if data['Total_Stops'].isnull().any():
        print("Warning: There are NaN values in the 'Total_Stops' column after mapping.")
    return data

def preprocess_data(raw):
    data = raw.copy()

    # Apply function to the column
    data['Duration'] = data['Duration'].apply(convert_to_minutes)

    # Preprocess date and time columns
    data = preprocess_date_and_time(data)

    # Handle categorical features
    categorical_columns = ['Airline', 'Source', 'Destination', 'Route', 'Additional_Info']
    data, label_encoders = encode_categorical_features(data, categorical_columns)

    # Normalize numerical columns
    numerical_columns = ['Journey_day', 'Journey_month', 'Dep_hour', 'Dep_minute', 'Arrival_hour', 'Arrival_minute', 'Duration']
    data, scaler = normalize_numerical_columns(data, numerical_columns)

    # Map Total_Stops
    data = map_total_stops(data)

    # Separate features and target variable
    X = data.drop(['Price', 'Date_of_Journey', 'Dep_Time', 'Arrival_Time'], axis=1)
    y = data['Price']

    return X, y, label_encoders, scaler