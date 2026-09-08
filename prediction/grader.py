"""
AgroGrow Quality Grading Module.
Classifies corn into Grade A, B, C, or D based on configurable quality thresholds,
assigns grading confidence, and generates summaries.
"""

from typing import Dict, Tuple
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class CornGrader:
    """
    Applies logic criteria to evaluate corn grade classifications.
    """
    @staticmethod
    def classify_grade(features: Dict[str, any], prediction_confidence: float) -> Dict[str, any]:
        """
        Calculates quality grade and builds classification summary.
        
        Args:
            features (Dict[str, any]): Map of extracted agronomic features.
            prediction_confidence (float): Mean classification confidence from predictor model.
            
        Returns:
            Dict[str, any]: Map containing Grade, Confidence, and Quality Summary.
        """
        healthy = features["healthy_percentage"]
        missing = features["missing_percentage"]
        disease = features["disease_percentage"]
        
        thresholds = global_config.grade_thresholds
        
        q_score = features.get("quality_score", healthy - 1.8 * disease - missing)
        
        # Grading rules logic evaluation (USDA / ISO aligned multi-criteria)
        if (healthy >= thresholds["A"]["min_healthy"] and 
            disease <= thresholds["A"]["max_disease"] and 
            missing <= thresholds["A"]["max_missing"]) or (q_score >= 82.0 and disease <= 4.5):
            grade = "Grade A"
            summary = ("Exceptional quality (Grade A / Premium Export). Corn kernels are highly uniform, "
                       "healthy, and fully suitable for premium commercial export markets.")
        elif (healthy >= thresholds["B"]["min_healthy"] and 
              disease <= thresholds["B"]["max_disease"] and 
              missing <= thresholds["B"]["max_missing"]) or (q_score >= 68.0 and disease <= 8.5):
            grade = "Grade B"
            summary = ("Good quality (Grade B / Standard Commercial). Satisfies standard export thresholds "
                       "with minimal kernel defects. Suitable for long-term storage under controlled conditions.")
        elif (healthy >= thresholds["C"]["min_healthy"] and 
              disease <= thresholds["C"]["max_disease"] and 
              missing <= thresholds["C"]["max_missing"]) or (q_score >= 45.0 and disease <= 18.0):
            grade = "Grade C"
            summary = ("Standard domestic grade (Grade C). Notable presence of missing or diseased kernels. "
                       "Recommended for domestic consumption, animal feed, or rapid processing.")
        else:
            grade = "Grade D"
            summary = ("Substandard quality (Grade D / Sample Grade). High concentration of diseased kernels "
                       "and/or severe kernel gaps. Unsafe for long-term storage; process immediately or inspect for mycotoxins.")
            
        # Overall grade confidence is a combination of classifier softmax confidence and grading margins
        # For simplicity, we directly utilize the model segmentation confidence
        grading_results = {
            "grade": grade,
            "confidence": prediction_confidence,
            "summary": summary
        }
        
        logger.info(f"Corn classified as: {grade} with confidence: {prediction_confidence:.4f}")
        return grading_results
