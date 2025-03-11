import numpy as np
import pandas as pd
import joblib
from utils import preprocess_data  # Assuming you have a preprocess function in utils.py

# Load the saved models
ensemble_model = joblib.load(r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\ensemble_model.pkl')
xgb_model = joblib.load(r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\xgb_model.pkl')
rf_model = joblib.load(r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\rf_model.pkl')
meta_model = joblib.load(r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\meta_model.pkl')

# Load the label encoders and scaler (if used during preprocessing)
label_encoders = joblib.load(r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\label_encoders.pkl')
scaler = joblib.load(r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\scaler.pkl')

# Function to preprocess new input data (same as used during training)
def preprocess_input_data(input_data, label_encoders, scaler):
    # Assume the preprocess_data function handles necessary preprocessing steps
    X_new, y, label_encoders, scaler = preprocess_data(input_data, label_encoders, scaler)
    return X_new

# Example of new input data (this would come from an external source, such as a user input, CSV, etc.)
new_data = pd.DataFrame({
   'Airline': ['IndiGo', 'Air India'],
   'Date_of_Journey': ['24/03/2019', '01/05/2019'],
   'Source': ['Banglore', 'Kolkata'],
   'Destination': ['New Delhi', 'Banglore'],
   'Route': ['BLR → DEL', 'CCU → IXR → BBI → BLR'],
   'Dep_Time': ['22:20', '05:50'],  
   'Arrival_Time': ['01:10 22 Mar', '13:15'],
   'Duration': ['2h 50m', '7h 25m'],
   'Total_Stops': ['non-stop', '2 stops'],
   'Additional_Info': ['No info','No info'],
   'Price': [0, 0] #place holder for target variable
})


# Preprocess the input data
X_new = preprocess_input_data(new_data, label_encoders, scaler)
print(X_new)

# Ensure the new data is the same shape as the data used in training
print("Input data shape:", X_new.shape)

# Make predictions using each of the models
y_pred_ensemble = ensemble_model.predict(X_new)
y_pred_xgb = xgb_model.predict(X_new)
y_pred_rf = rf_model.predict(X_new)

# Stack predictions to create input for the meta model
X_meta_new = np.column_stack([y_pred_ensemble, y_pred_xgb, y_pred_rf])

# Make final prediction using the meta model
y_pred_meta = meta_model.predict(X_meta_new)

# Output the prediction (e.g., predicted ticket price)
print(f"Predicted Ticket Price: {y_pred_meta}")

##the results do not seem to be correct this is due to over fitting to flight duration 