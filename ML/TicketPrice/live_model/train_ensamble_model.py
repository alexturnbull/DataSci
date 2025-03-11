import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
import kagglehub
import joblib
import matplotlib.pyplot as plt

# Import preprocessing and ModelEnsembler class
from utils import preprocess_data
from model_class import ModelEnsembler

# Load dataset
path = kagglehub.dataset_download("ibrahimelsayed182/plane-ticket-price")
print("Path to dataset files:", path)
data_path = fr'{path}\ticket_pricing_data.csv'
raw = pd.read_csv(r'C:\Users\Alex\.cache\kagglehub\datasets\ibrahimelsayed182\plane-ticket-price\versions\1\Data_Train.csv')

# Process data using the preprocess_data function from utils.py
X, y, label_encoders, scaler = preprocess_data(raw)

# Save label encoders and scaler locally
joblib.dump(label_encoders, r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\label_encoders.pkl')
joblib.dump(scaler, r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and train the ModelEnsembler
input_shape = (X_train.shape[1],)
ensembler = ModelEnsembler(input_shape)

# Train all models and get the final trained meta-model
meta_model = ensembler(X_train, y_train, X_test, y_test)

#Ensure we use all 3 models for meta-model predictions**
X_meta = np.column_stack([
    ensembler.ensemble_model.predict(X_test),  # Neural Network ensemble
    ensembler.xgb_model.predict(X_test),       # XGBoost
    ensembler.rf_model.predict(X_test)         # Random Forest
])

y_pred_meta = meta_model.predict(X_meta)

# Evaluate performance
mae_stacked = mean_absolute_error(y_test, y_pred_meta)
mse_stacked = mean_squared_error(y_test, y_pred_meta)
rmse_stacked = np.sqrt(mse_stacked)
r2_stacked = r2_score(y_test, y_pred_meta)
print(f"Model Accuracy: {meta_model.score(X_meta, y_test)}")
print(f"MAE: {mae_stacked}")
print(f"RMSE: {rmse_stacked}")
print(f"R² Score: {r2_stacked}")
print(f"MAE of Stacked Model: {mae_stacked}")

residuals = y_test - y_pred_meta

plt.figure(figsize=(10, 6))
plt.scatter(y_pred_meta, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

# # Plot predictions vs actual values
# plt.scatter(X_test['Duration'], y_test, color='black', label='Actual')
# plt.scatter(X_test['Duration'], y_pred_meta, color='red', label='Predicted')
# plt.xlabel('Duration (minutes)')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

importances = ensembler.rf_model.feature_importances_
features = X_train.columns

plt.barh(features, importances)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Feature Importance from Random Forest")
plt.show()

# Save meta-model locally
joblib.dump(ensembler.ensemble_model, r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\ensemble_model.pkl')
joblib.dump(ensembler.xgb_model, r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\xgb_model.pkl')
joblib.dump(ensembler.rf_model, r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\rf_model.pkl')
joblib.dump(meta_model, r'C:\Projects\datasci2\DataSci\ML\TicketPrice\live_model\model_outputs\meta_model.pkl')

print("Models saved successfully!")
