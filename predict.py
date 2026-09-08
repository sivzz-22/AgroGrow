"""
AgroGrow Prediction CLI Entrypoint.
Runs the complete post-harvest quality assessment pipeline on an image or folder,
grades it, predicts storage life, and generates a PDF report.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger
from AgroGrow.prediction.predictor import CornPredictor
from AgroGrow.prediction.feature_extractor import CornFeatureExtractor
from AgroGrow.prediction.grader import CornGrader
from AgroGrow.storage_prediction.model import StoragePredictor
from AgroGrow.utils.report_generator import PDFReportGenerator

def run_pipeline(
    image_path: Path,
    temperature: float = 25.0,
    humidity: float = 70.0,
    storage_type: str = "Open Air"
) -> dict:
    """
    Executes the full assessment pipeline for a single image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")
        
    # 1. Image Segmentation Inference
    predictor = CornPredictor()
    mask, overlay, _, confidence = predictor.predict_single(image_path)
    
    # Save the overlay image temporarily or in the results folder
    overlay_path = global_config.results_dir / f"{image_path.stem}_overlay.png"
    import cv2
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved overlay image to {overlay_path}")
    
    # 2. Feature Extraction
    features = CornFeatureExtractor.extract_features(mask)
    
    # 3. Quality Grading
    grading = CornGrader.classify_grade(features, confidence)
    
    # 4. Storage Prediction
    try:
        storage_pred = StoragePredictor()
        shelf_life, risk_level, rec = storage_pred.predict(
            healthy_pct=features["healthy_percentage"],
            disease_pct=features["disease_percentage"],
            missing_pct=features["missing_percentage"],
            defect_area=features["defect_area_pixels"],
            corn_area=features["total_corn_area_pixels"],
            temperature=temperature,
            humidity=humidity,
            storage_type=storage_type
        )
    except Exception as e:
        logger.warning(f"Storage predictor not trained or failed: {e}. Falling back to default heuristics.")
        # Fallback heuristic
        shelf_life = 45.0
        risk_level = "Medium Risk"
        rec = "Default recommendation: Ensure low moisture and cool temperature ventilation."
        
    storage_results = {
        "shelf_life_days": shelf_life,
        "risk_level": risk_level,
        "recommendation": rec,
        "temperature_c": temperature,
        "humidity_pct": humidity,
        "storage_type": storage_type
    }
    
    # 5. Report Generation
    pdf_path = global_config.reports_dir / f"{image_path.stem}_quality_report.pdf"
    PDFReportGenerator.generate(
        image_path=image_path,
        overlay_path=overlay_path,
        features=features,
        grading=grading,
        storage=storage_results,
        output_pdf_path=pdf_path
    )
    
    return {
        "pdf_report": pdf_path,
        "overlay_image": overlay_path,
        "features": features,
        "grading": grading,
        "storage": storage_results
    }

def main():
    parser = argparse.ArgumentParser(description="AgroGrow Corn Quality Assessment Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path to input corn image (.jpg)")
    parser.add_argument("--temp", type=float, default=25.0, help="Storage temperature in Celsius")
    parser.add_argument("--humidity", type=float, default=70.0, help="Storage relative humidity in percentage")
    parser.add_argument("--storage", type=str, default="Open Air", choices=global_config.storage_types, help="Type of storage container")
    parser.add_argument("--query", type=str, default="Is this batch suitable for export?", help="Question for the AI Assistant")
    
    args = parser.parse_args()
    
    try:
        results = run_pipeline(
            image_path=Path(args.image),
            temperature=args.temp,
            humidity=args.humidity,
            storage_type=args.storage,
            assistant_query=args.query
        )
        
        # Display summary in stdout
        print("\n" + "="*50)
        print(" AGROGROW CORN QUALITY ASSESSMENT SUMMARY")
        print("="*50)
        print(f"File Name: {Path(args.image).name}")
        print(f"Quality Grade: {results['grading']['grade']} (Confidence: {results['grading']['confidence']:.2%})")
        print(f"Quality Score: {results['features']['quality_score']:.1f}/100")
        print(f"Healthy Kernels: {results['features']['healthy_percentage']:.2f}%")
        print(f"Diseased Kernels: {results['features']['disease_percentage']:.2f}%")
        print(f"Missing Kernels: {results['features']['missing_percentage']:.2f}%")
        print("-" * 50)
        print(f"Predicted Storage Life: {results['storage']['shelf_life_days']:.1f} Days")
        print(f"Storage Risk Category: {results['storage']['risk_level']}")
        print(f"Report PDF Saved to: {results['pdf_report']}")
        print("="*50)
        print(f"AI Assistant Answer to '{args.query}':\n{results['assistant_reply']}")
        print("="*50 + "\n")
        
    except Exception as e:
        logger.exception(f"Pipeline execution encountered an error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
