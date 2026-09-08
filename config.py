import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple

# Base Project Root
PROJECT_ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    # General Paths
    project_root: Path = PROJECT_ROOT
    raw_dataset_dir: Path = Path("D:/Sem_7/research paper 1/DataSet")
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    images_dir: Path = dataset_dir / "images"
    masks_dir: Path = dataset_dir / "masks"
    
    weights_dir: Path = PROJECT_ROOT / "weights"
    results_dir: Path = PROJECT_ROOT / "results"
    reports_dir: Path = PROJECT_ROOT / "reports"
    docs_dir: Path = PROJECT_ROOT / "docs"
    
    # Model Configurations
    input_size: Tuple[int, int] = (512, 512)  # Height, Width (Matches Research Paper Resolution)
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["background", "healthy", "missing", "diseased"])
    class_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (0, 0, 0),        # Background: Black
        (0, 255, 0),      # Healthy: Green
        (0, 0, 255),      # Missing: Blue
        (255, 0, 0)       # Diseased: Red
    ])
    
    # Training Hyperparameters
    batch_size: int = 2
    learning_rate: float = 1e-4
    epochs: int = 200
    early_stopping_patience: int = 20
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    weight_decay: float = 1e-4
    device: str = "cuda"  # Will be dynamically validated and set to cpu if cuda is unavailable
    
    # Loss Coefficients
    wce_weight: float = 0.5
    dice_weight: float = 0.5
    class_weights: List[float] = field(default_factory=lambda: [0.05, 1.0, 8.0, 4.0])  # Model 4: heavily upweight Missing (Class 2) since it is only ~2% of pixels
    
    # Inference / Feature Extraction Parameters
    kernel_density_threshold: float = 0.5
    min_corn_area_pixels: int = 1000
    
    # Quality Grading Criteria (Aligned with USDA / ISO agricultural standards)
    grade_thresholds: dict = field(default_factory=lambda: {
        "A": {"min_healthy": 88.0, "max_disease": 4.0, "max_missing": 4.0},
        "B": {"min_healthy": 75.0, "max_disease": 8.0, "max_missing": 10.0},
        "C": {"min_healthy": 55.0, "max_disease": 16.0, "max_missing": 18.0}
    })
    
    # Storage Types
    storage_types: List[str] = field(default_factory=lambda: ["Open Air", "Cold Storage", "Hermetic Bag"])
    
    # AI Assistant Setup
    ollama_model: str = "llama3"
    openai_api_model: str = "gpt-4-turbo"

    def __post_init__(self):
        # Create directories if they do not exist
        for directory in [self.dataset_dir, self.images_dir, self.masks_dir, 
                          self.weights_dir, self.results_dir, self.reports_dir, self.docs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

# Instantiate the global config
global_config = Config()
