"""
AgroGrow Dataset Preparation and Validation Entrypoint.
Initializes folder structure, copies raw images, generates pseudo-masks, and validates.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.utils.prepare_dataset import DatasetPreparer
from AgroGrow.utils.validation import DatasetValidator

def main():
    logger.info("Initializing AgroGrow dataset preparation and validation flow...")
    
    # Instantiate preparer with raw data path and target dataset path
    preparer = DatasetPreparer(
        raw_dir=global_config.raw_dataset_dir,
        target_dir=global_config.dataset_dir
    )
    
    # Process files
    preparer.process()
    
    # Instantiate validator and run checks
    validator = DatasetValidator(dataset_dir=global_config.dataset_dir)
    success, message = validator.validate()
    
    if success:
        logger.info(f"Dataset successfully prepared and validated! Report saved to: {message}")
    else:
        logger.error(f"Dataset preparation failed during validation: {message}")
        raise RuntimeError(message)

if __name__ == "__main__":
    main()
