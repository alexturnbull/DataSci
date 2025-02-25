import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import kagglehub
import joblib
from utils import preprocess_data

# Load dataset (example: historical ticket pricing data)
path = kagglehub.dataset_download("ibrahimelsayed182/plane-ticket-price")
print("Path to dataset files:", path)
data_path = fr'{path}\ticket_pricing_data.csv'
raw = pd.read_csv(r'C:\Users\Alex\.cache\kagglehub\datasets\ibrahimelsayed182\plane-ticket-price\versions\1\Data_Train.csv')

#process data using the preprocess_data function from utils.py
X, y, label_encoders, scaler = preprocess_data(raw)

#save label encoders and scaler locally

joblib.dump(label_encoders, r'C:\Projects\DataSci\ML\TicketPrice\live_model\model_outputs\label_encoders.pkl')
joblib.dump(scaler, r'C:\Projects\DataSci\ML\TicketPrice\live_model\model_outputs\scaler.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_model_1(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output price
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model

def create_model_2(input_shape):
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output price
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def create_model_3(input_shape):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.Conv1D(64, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output price
    ])
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


# Train Random Forest
def create_rf_model():
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    return rf_model

# Train XGBoost
def create_xgb_model():
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    return xgb_model


input_shape = (X_train.shape[1],)  # Shape of the input features

model1_nn = create_model_1(input_shape)
model2_nn = create_model_2(input_shape)
model3_nn = create_model_3(input_shape)

models_list = [model1_nn, model2_nn, model3_nn]
for i, model in enumerate(models_list):
    print(f"Training Model {i+1}...")
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))


def averaging_ensemble(models, input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    outputs = [model(inputs) for model in models]
    avg_output = tf.keras.layers.Average()(outputs)
    ensemble_model = tf.keras.Model(inputs=inputs, outputs=avg_output)
    return ensemble_model

ensemble_avg = averaging_ensemble(models_list, input_shape)
ensemble_avg.compile(optimizer='adam', loss='mean_absolute_error')


y_pred_nn_avg = ensemble_avg.predict(X_test)

mae_avg = mean_absolute_error(y_test, y_pred_nn_avg)

print(f"MAE of Averaging Ensemble: {mae_avg}")

#add random forest and xgboost models to the ensemble
rf_model = create_rf_model()
xgb_model = create_xgb_model()

# Train Random Forest
rf_model.fit(X_train, y_train)

# Train XGBoost
xgb_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test).reshape(-1, 1)
y_pred_xgb = xgb_model.predict(X_test).reshape(-1, 1)

##meta model 

meta_predictions = np.mean(np.column_stack((y_pred_nn_avg, y_pred_rf, y_pred_xgb)), axis=1)
X_meta = np.column_stack((y_pred_nn_avg, y_pred_xgb, y_pred_rf))

# Train a meta-model on the stacked predictions
meta_model = LinearRegression()
meta_model.fit(X_meta, y_test)

# You can now use this meta-model to make final predictions on new data
final_predictions = meta_model.predict(X_meta)

# Evaluate the performance of the stacked model
mae_stacked = mean_absolute_error(y_test, final_predictions)
print(f"MAE of Stacked Model: {mae_stacked}")
print(f"Model accuracy: {meta_model.score(X_meta, y_test)}")

import matplotlib.pyplot as plt
plt.scatter(X_test['Duration'], y_test, color='black', label='Actual')
plt.scatter(X_test['Duration'], final_predictions, color='red', label='Predicted')
plt.xlabel('Duration (minutes)')
plt.ylabel('Price')
plt.legend()
plt.show()
#save meta model locally

joblib.dump(meta_model, r'C:\Projects\DataSci\ML\TicketPrice\live_model\model_outputs\meta_model.pkl')
