"""
AgroGrow Storage Life Data Generator.
Synthesizes simulated storage shelf-life data based on agricultural physics
to train the Random Forest and XGBoost regression models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class StorageDataGenerator:
    """
    Simulates environmental factors and crop metrics to predict shelf life.
    """
    def __init__(self, num_samples: int = 1200, seed: int = 42):
        self.num_samples = num_samples
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        """
        Generates simulated crop storage data.
        
        Formula:
            Base days by storage type:
                - Cold Storage: 180
                - Hermetic Bag: 120
                - Open Air: 35
            Degradation multipliers:
                - Temp factor: exp(-0.05 * (temp - 10))
                - Humid factor: exp(-0.02 * (humidity - 55))
                - Quality factor: (healthy_pct / 100)^2 * exp(-3.0 * (disease_pct / 100))
        """
        np.random.seed(self.seed)
        
        # 1. Generate quality features
        healthy_pct = np.random.uniform(50.0, 98.0, self.num_samples)
        disease_pct = np.random.uniform(0.0, 100.0 - healthy_pct)
        missing_pct = 100.0 - healthy_pct - disease_pct
        
        corn_area = np.random.uniform(8000, 22000, self.num_samples)
        defect_area = (disease_pct + missing_pct) / 100.0 * corn_area
        
        # 2. Generate environmental features tailored by storage type
        storage_types = np.random.choice(global_config.storage_types, self.num_samples)
        
        temperatures = []
        for st_type in storage_types:
            if st_type == "Cold Storage":
                # Cold storage operates in sub-zero and chilled ranges (-20°C to +10°C)
                temperatures.append(np.random.uniform(-20.0, 10.0))
            elif st_type == "Hermetic Bag":
                temperatures.append(np.random.uniform(5.0, 40.0))
            else:  # Open Air
                temperatures.append(np.random.uniform(10.0, 48.0))
        temperature = np.array(temperatures)
        humidity = np.random.uniform(30.0, 95.0, self.num_samples)
        
        # 3. Compute target shelf life
        shelf_life_days = []
        for i in range(self.num_samples):
            st_type = storage_types[i]
            temp = temperature[i]
            hum = humidity[i]
            h_pct = healthy_pct[i]
            d_pct = disease_pct[i]
            
            # Base shelf life
            if st_type == "Cold Storage":
                base_life = 260.0
                # In cold storage, sub-zero temperatures preserve kernels with minimal enzymatic decay
                eff_temp = max(-15.0, temp)
                temp_factor = np.exp(-0.035 * (eff_temp - 0.0))
            elif st_type == "Hermetic Bag":
                base_life = 130.0
                eff_temp = temp
                temp_factor = np.exp(-0.04 * (eff_temp - 15.0))
            else:  # Open Air
                base_life = 40.0
                eff_temp = temp
                temp_factor = np.exp(-0.045 * (eff_temp - 18.0))
            
            # Humidity factor (optimal is 55% RH or below, higher humidity increases mold)
            hum_factor = np.exp(-0.015 * (hum - 55.0))
            
            # Quality factor
            quality_factor = ((h_pct / 100.0) ** 1.8) * np.exp(-3.0 * (d_pct / 100.0))
            
            # Combined shelf life
            life = base_life * temp_factor * hum_factor * quality_factor
            
            # Add Gaussian noise
            noise = np.random.normal(0, 0.05 * life)
            final_life = max(3.0, min(365.0, life + noise))
            shelf_life_days.append(final_life)
            
        df = pd.DataFrame({
            "healthy_pct": healthy_pct,
            "disease_pct": disease_pct,
            "missing_pct": missing_pct,
            "defect_area_pixels": defect_area,
            "corn_area_pixels": corn_area,
            "temperature_c": temperature,
            "humidity_pct": humidity,
            "storage_type": storage_types,
            "shelf_life_days": shelf_life_days
        })
        
        # Save to dataset folder
        csv_path = global_config.dataset_dir / "storage_data.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Generated {self.num_samples} storage records saved to {csv_path}")
        return df

if __name__ == "__main__":
    generator = StorageDataGenerator()
    generator.generate()
