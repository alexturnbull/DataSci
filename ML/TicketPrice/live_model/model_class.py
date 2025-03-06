import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error

class ModelEnsembler:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.models = []
        self.rf_model = None
        self.xgb_model = None
        self.ensemble_model = None
        self.meta_model = None

    def create_simple_dense_model(self):
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=self.input_shape),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def create_regularized_dense_model(self):
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=self.input_shape),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def create_cnn_model(self):
        model = models.Sequential([
            layers.Reshape((self.input_shape[0], 1), input_shape=self.input_shape),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(128, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_absolute_error')
        return model

    def create_rf_model(self):
        return RandomForestRegressor(n_estimators=100, random_state=42)

    def create_xgb_model(self):
        return XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    def train_neural_networks_kfold(self, X, y, epochs=10, batch_size=64, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]  # Correct indexing with .iloc
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]  # Correct indexing with .iloc

            print(f"Training Neural Network Models for Fold {fold+1}...")

            models_for_fold = [
                self.create_simple_dense_model(),
                self.create_regularized_dense_model(),
                self.create_cnn_model()
            ]

            fold_predictions = []

            for i, model in enumerate(models_for_fold):
                model.fit(X_train_fold, y_train_fold, epochs=epochs, batch_size=batch_size, validation_data=(X_val_fold, y_val_fold), verbose=1)
                fold_pred = model.predict(X_val_fold)
                fold_predictions.append(fold_pred)

            # Average the predictions for ensemble
            avg_predictions = np.mean(fold_predictions, axis=0)
            mae = mean_absolute_error(y_val_fold, avg_predictions)
            fold_scores.append(mae)

            # Add models to self.models for ensemble later
            self.models.extend(models_for_fold)  # Appending models to self.models

        avg_mae = np.mean(fold_scores)
        print(f"Average MAE after {n_splits}-Fold Cross-Validation: {avg_mae}")

    def build_averaging_ensemble(self):
        inputs = tf.keras.Input(shape=self.input_shape)
        outputs = [model(inputs) for model in self.models]
        avg_output = tf.keras.layers.Average()(outputs)
        self.ensemble_model = tf.keras.Model(inputs=inputs, outputs=avg_output)
        self.ensemble_model.compile(optimizer='adam', loss='mean_absolute_error')

    def train_sklearn_models_kfold(self, X, y, n_splits=5):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        rf_fold_scores = []
        xgb_fold_scores = []

        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            # Correct indexing with .iloc
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

            print(f"Training Random Forest Model for Fold {fold+1}...")
            rf_model = self.create_rf_model()
            rf_model.fit(X_train_fold, y_train_fold)
            rf_pred = rf_model.predict(X_val_fold)
            rf_mae = mean_absolute_error(y_val_fold, rf_pred)
            rf_fold_scores.append(rf_mae)

            print(f"Training XGBoost Model for Fold {fold+1}...")
            xgb_model = self.create_xgb_model()
            xgb_model.fit(X_train_fold, y_train_fold)
            xgb_pred = xgb_model.predict(X_val_fold)
            xgb_mae = mean_absolute_error(y_val_fold, xgb_pred)
            xgb_fold_scores.append(xgb_mae)

        print(f"Random Forest MAE across folds: {np.mean(rf_fold_scores)} ± {np.std(rf_fold_scores)}")
        print(f"XGBoost MAE across folds: {np.mean(xgb_fold_scores)} ± {np.std(xgb_fold_scores)}")

    def train_meta_model_kfold(self, X, y, n_splits=5):
        if not self.ensemble_model:
            raise ValueError("Ensemble model is not trained. Call 'build_averaging_ensemble()' and train first.")
    
        if not self.rf_model:
            print("Training Random Forest model...")
            self.rf_model = self.create_rf_model()
            self.rf_model.fit(X, y)  # Train the Random Forest model
    
        if not self.xgb_model:
            print("Training XGBoost model...")
            self.xgb_model = self.create_xgb_model()
            self.xgb_model.fit(X, y)  # Train the XGBoost model
    
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        meta_predictions = []
        meta_targets = []
    
        for fold, (train_index, val_index) in enumerate(kf.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
    
            # Get the predictions from the models and reshape to (n_samples, 1)
            y_pred_nn_avg = self.ensemble_model.predict(X_val_fold).reshape(-1, 1)  # Reshape here
            y_pred_rf = self.rf_model.predict(X_val_fold).reshape(-1, 1)  # Reshape here
            y_pred_xgb = self.xgb_model.predict(X_val_fold).reshape(-1, 1)  # Reshape here
    
            # Combine the predictions into features for the meta-model
            fold_meta_features = np.hstack((y_pred_nn_avg, y_pred_xgb, y_pred_rf))
    
            # Train the meta-model (Linear Regression)
            meta_model = LinearRegression()
            meta_model.fit(fold_meta_features, y_val_fold)
    
            # Make predictions for the fold
            fold_meta_predictions = meta_model.predict(fold_meta_features)
    
            # Append predictions and targets for later averaging
            meta_predictions.append(fold_meta_predictions.flatten())
            meta_targets.append(y_val_fold)
    
        # Convert meta_predictions and meta_targets to NumPy arrays
        meta_predictions = np.concatenate(meta_predictions)
        meta_targets = np.concatenate(meta_targets)
    
        # Calculate MAE for the meta-model
        mae = mean_absolute_error(meta_targets, meta_predictions)
        print(f"Meta-model MAE after {n_splits}-Fold Cross-Validation: {mae}")
    
        # Train the final meta-model on the full dataset
        y_pred_nn_avg_full = self.ensemble_model.predict(X).reshape(-1, 1)
        y_pred_rf_full = self.rf_model.predict(X).reshape(-1, 1)
        y_pred_xgb_full = self.xgb_model.predict(X).reshape(-1, 1)
        meta_features_full = np.hstack((y_pred_nn_avg_full, y_pred_xgb_full, y_pred_rf_full))
    
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features_full, y)




    def get_final_model(self):
        """Returns the trained meta-model after full training."""
        if self.meta_model is None:
            raise ValueError("Meta-model is not trained yet. Train models first by calling `train_meta_model_kfold`.")
        return self.meta_model

    def __call__(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=64, n_splits=5):
     """Runs the full pipeline and returns the final trained meta-model."""
     print("Starting K-Fold training pipeline...")

     # Train neural networks (add models to self.models)
     self.train_neural_networks_kfold(X_train, y_train, epochs, batch_size, n_splits)

     # Train sklearn models (Random Forest & XGBoost)
     self.train_sklearn_models_kfold(X_train, y_train, n_splits)

     # Build the ensemble model by averaging the predictions of the neural networks
     self.build_averaging_ensemble()  # This will work now, since self.models is populated

     # Train the meta-model using the ensemble predictions
     self.train_meta_model_kfold(X_test, y_test, n_splits)

     print("K-Fold training pipeline completed successfully!")
     return self.get_final_model()

