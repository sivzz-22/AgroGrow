"""
AgroGrow Storage Life Model Module.
Trains, evaluates, and compares Random Forest and XGBoost regressors
for storage shelf-life prediction, and provides an inference predictor.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from typing import Dict, Tuple

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class StoragePredictor:
    """
    Loads saved model and preprocessing parameters to run shelf life inference.
    """
    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir if model_dir is not None else global_config.weights_dir
        self.model_path = self.model_dir / "best_storage_model.pkl"
        
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        if self.model_path.exists():
            self.load()

    def save(self, model: any, scaler: StandardScaler, feature_columns: list):
        """Saves model state and metadata."""
        state = {
            "model": model,
            "scaler": scaler,
            "feature_columns": feature_columns
        }
        with open(self.model_path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved best storage prediction model to {self.model_path}")

    def load(self):
        """Loads model state and metadata."""
        with open(self.model_path, "rb") as f:
            state = pickle.load(f)
        self.model = state["model"]
        self.scaler = state["scaler"]
        self.feature_columns = state["feature_columns"]
        logger.info("Storage predictor weights successfully loaded.")

    def predict(
        self,
        healthy_pct: float,
        disease_pct: float,
        missing_pct: float,
        defect_area: float,
        corn_area: float,
        temperature: float,
        humidity: float,
        storage_type: str
    ) -> Tuple[float, str, str]:
        """
        Runs inference on storage conditions and crop metrics.
        
        Returns:
            Tuple[float, str, str]: (predicted_days, risk_level, recommendation)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model is not trained or loaded.")
            
        # 1. Build dictionary matching the training columns
        input_data = {
            "healthy_pct": healthy_pct,
            "disease_pct": disease_pct,
            "missing_pct": missing_pct,
            "defect_area_pixels": defect_area,
            "corn_area_pixels": corn_area,
            "temperature_c": temperature,
            "humidity_pct": humidity
        }
        
        # 2. Add one-hot encoded storage types
        for st in global_config.storage_types:
            input_data[f"storage_type_{st}"] = 1.0 if st == storage_type else 0.0
            
        # 3. Format as DataFrame with identical columns sequence
        df_input = pd.DataFrame([input_data])
        df_input = df_input[self.feature_columns]
        
        # 4. Scale inputs and run prediction
        scaled_input = self.scaler.transform(df_input)
        pred_days = float(self.model.predict(scaled_input)[0])
        pred_days = max(1.0, pred_days)  # Clamp to at least 1 day
        
        # 5. Determine Risk Level
        if pred_days > 90:
            risk_level = "Low Risk"
        elif pred_days >= 30:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        # 6. Formulate Recommendation
        if risk_level == "High Risk":
            rec = ("CRITICAL: High risk of rapid spoilage or mold infestation. "
                   "Reduce humidity and lower storage temperature immediately. "
                   "Recommended for immediate local processing, not suitable for export.")
        elif risk_level == "Medium Risk":
            rec = ("WARNING: Moderate storage stability. Keep moisture low (below 14%) and "
                   "ensure proper aeration. Suitable for domestic markets with active monitoring.")
        else:
            rec = ("STABLE: High quality storage profile. Suitable for medium to long-term storage "
                   "under hermetic sealing or cold ventilation. Ideal for export markets.")
            
        return pred_days, risk_level, rec


def train_storage_models() -> Dict[str, any]:
    """
    Loads simulated storage dataset, trains Random Forest & XGBoost regressors,
    compares them, plots comparison bar chart, and saves the best model.
    """
    csv_path = global_config.dataset_dir / "storage_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Storage data not found at {csv_path}. Run generator first.")
        
    df = pd.read_csv(csv_path)
    
    # 1. Feature Preprocessing (One-Hot encode storage type)
    df_encoded = pd.get_dummies(df, columns=["storage_type"])
    
    # Extract target and features
    X = df_encoded.drop(columns=["shelf_life_days"])
    y = df_encoded["shelf_life_days"]
    
    feature_cols = list(X.columns)
    
    # 2. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Random Forest
    logger.info("Training Random Forest Regressor for shelf-life prediction...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    
    # 5. Train XGBoost
    logger.info("Training XGBoost Regressor for shelf-life prediction...")
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    xgb_preds = xgb_model.predict(X_test_scaled)
    
    # 6. Evaluate and Compare
    metrics = {}
    for name, preds in [("Random Forest", rf_preds), ("XGBoost", xgb_preds)]:
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        metrics[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}
        logger.info(f"{name} Results - R2: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
    # Choose best model (based on R2 score)
    best_name = "Random Forest" if metrics["Random Forest"]["R2"] > metrics["XGBoost"]["R2"] else "XGBoost"
    best_model = rf_model if best_name == "Random Forest" else xgb_model
    logger.info(f"Selected best storage prediction model: {best_name}")
    
    # Save the best model
    predictor = StoragePredictor()
    predictor.save(best_model, scaler, feature_cols)
    
    # 7. Generate comparison plot
    plt.figure(figsize=(10, 5))
    x_axis = np.arange(3)
    width = 0.35
    
    rf_metrics = [metrics["Random Forest"]["R2"], metrics["Random Forest"]["RMSE"], metrics["Random Forest"]["MAE"]]
    xgb_metrics = [metrics["XGBoost"]["R2"], metrics["XGBoost"]["RMSE"], metrics["XGBoost"]["MAE"]]
    
    plt.bar(x_axis - width/2, rf_metrics, width, label="Random Forest", color="teal")
    plt.bar(x_axis + width/2, xgb_metrics, width, label="XGBoost", color="coral")
    
    plt.xticks(x_axis, ["R2 Score", "RMSE (Days)", "MAE (Days)"])
    plt.title("Shelf-Life Prediction Model Performance Comparison", fontsize=14, fontweight="bold")
    plt.ylabel("Metric Value")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    
    plot_path = global_config.results_dir / "storage_model_comparison.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
    plt.close()
    
    return metrics
