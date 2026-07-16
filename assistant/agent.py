"""
AgroGrow AI Assistant Module.
Provides natural-language explanations and answers queries regarding corn quality,
defects, grading decisions, storage life, and export suitability.
"""

import os
import re
import requests
from typing import Dict, Tuple
from AgroGrow.config import global_config
from AgroGrow.utils.logger import logger

class CornAssistant:
    """
    An expert system assistant combined with optional OpenAI API LLM capabilities.
    """
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        if self.openai_key:
            logger.info("OpenAI API key detected. AI Assistant will run in LLM mode.")
        else:
            logger.info("No OpenAI API key found. AI Assistant will run in expert rule-based mode.")

    def _query_openai(self, prompt: str) -> str:
        """Sends a prompt to OpenAI API."""
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_key}"
        }
        data = {
            "model": global_config.openai_api_model,
            "messages": [
                {"role": "system", "content": "You are AgroGrow-Bot, an expert AI agronomist specializing in post-harvest corn quality, grain grading, and storage predictions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.5
        }
        try:
            response = requests.post(url, json=data, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                logger.error(f"OpenAI API call failed: {response.text}")
                return "Error calling OpenAI API. Falling back to local rules."
        except Exception as e:
            logger.error(f"OpenAI connection error: {e}")
            return "Connection error to OpenAI. Falling back to local rules."

    def answer_query(self, query: str, context: Dict[str, any]) -> str:
        """
        Answers a user question given the analysis context.
        
        Args:
            query (str): The question asked by the user.
            context (Dict[str, any]): Dictionary containing:
                - healthy_pct, disease_pct, missing_pct, defect_area_pixels, total_corn_area_pixels
                - grade, confidence, summary (from grader)
                - shelf_life_days, risk_level, recommendation (from storage prediction)
                - temperature_c, humidity_pct, storage_type (current user environmental selections)
                
        Returns:
            str: Natural-language response.
        """
        query_lower = query.lower()
        
        # Check if we should use LLM mode
        if self.openai_key:
            prompt = f"""
            Analyze the following corn quality metrics and answer the user's question.
            
            Metrics Context:
            - Quality Grade: {context.get('grade', 'N/A')} (Confidence: {context.get('confidence', 0.0):.2%})
            - Grade Summary: {context.get('summary', 'N/A')}
            - Healthy Kernels: {context.get('healthy_pct', 0.0):.2f}%
            - Diseased Kernels: {context.get('disease_pct', 0.0):.2f}%
            - Missing Kernels: {context.get('missing_pct', 0.0):.2f}%
            - Total Corn Area: {context.get('total_corn_area_pixels', 0)} pixels
            - Defect Area: {context.get('defect_area_pixels', 0)} pixels
            - Predicted Shelf Life: {context.get('shelf_life_days', 0.0):.1f} days
            - Storage Risk Level: {context.get('risk_level', 'N/A')}
            - Storage Recommendation: {context.get('recommendation', 'N/A')}
            - Simulated Environmental Conditions: Temperature {context.get('temperature_c', 25.0)}°C, Humidity {context.get('humidity_pct', 70.0)}% RH, Storage Type '{context.get('storage_type', 'Open Air')}'
            
            User Question: "{query}"
            
            Provide a clear, detailed, and professional explanation based strictly on the metrics.
            """
            response = self._query_openai(prompt)
            if "Falling back" not in response:
                return response
                
        # Rule-based fallback
        h_pct = context.get('healthy_pct', 0.0)
        d_pct = context.get('disease_pct', 0.0)
        m_pct = context.get('missing_pct', 0.0)
        grade = context.get('grade', 'Grade D')
        shelf_life = context.get('shelf_life_days', 0.0)
        risk = context.get('risk_level', 'High Risk')
        rec = context.get('recommendation', '')
        st_type = context.get('storage_type', 'Open Air')
        temp = context.get('temperature_c', 25.0)
        hum = context.get('humidity_pct', 70.0)

        # 1. Why is this corn Grade X?
        if "why" in query_lower and "grade" in query_lower:
            return (
                f"This corn is classified as **{grade}** because of its kernel ratio. "
                f"Specifically, it contains **{h_pct:.1f}% healthy kernels**, "
                f"**{d_pct:.1f}% diseased kernels**, and **{m_pct:.1f}% missing kernels**.\n\n"
                f"Under the grading rules:\n"
                f"- **Grade A** requires healthy ≥ 90%, disease = 0%, and missing ≤ 2%.\n"
                f"- **Grade B** requires healthy ≥ 75%, disease ≤ 3%, and missing ≤ 8%.\n"
                f"- **Grade C** requires healthy ≥ 50%, disease ≤ 10%, and missing ≤ 15%.\n"
                f"Since your corn metrics fail to meet the upper tiers but fit this category, it falls into {grade}."
            )
            
        # 2. What caused quality reduction?
        elif "reduction" in query_lower or "cause" in query_lower or "defect" in query_lower:
            defects = []
            if d_pct > 0:
                defects.append(f"diseased kernels ({d_pct:.1f}%)")
            if m_pct > 0:
                defects.append(f"missing kernels/gaps ({m_pct:.1f}%)")
            defect_str = " and ".join(defects) if defects else "no notable defects"
            return (
                f"The quality reduction is primarily caused by **{defect_str}**. "
                f"This accounts for a total defect area of **{context.get('defect_area_pixels', 0)} pixels**, "
                f"meaning that {(d_pct+m_pct):.1f}% of the corn area is damaged, reducing the overall quality score."
            )
            
        # 3. How long can it be stored?
        elif "long" in query_lower or "shelf life" in query_lower or "stored" in query_lower:
            return (
                f"This batch is estimated to have a shelf life of **{shelf_life:.1f} days** "
                f"under {st_type} storage at {temp}°C and {hum}% relative humidity.\n\n"
                f"This indicates a **{risk}** level. Fungal development and kernel degradation "
                f"will accelerate if storage moisture and heat are not strictly managed."
            )
            
        # 4. Storage recommendations
        elif "condition" in query_lower or "recommend" in query_lower or "store" in query_lower:
            return (
                f"For this batch, we recommend: **{rec}**\n\n"
                f"General grain science guidelines suggest drying the corn to a moisture content below 14%, "
                f"maintaining a cool ventilation environment (preferably cold storage at 5-10°C), "
                f"and using hermetic bags to limit oxygen availability, preventing mold growth."
            )
            
        # 5. Export suitability
        elif "export" in query_lower or "suitable" in query_lower:
            if grade in ["Grade A", "Grade B"]:
                return (
                    f"Yes, this batch is **suitable for export** as it is graded as **{grade}**. "
                    f"The healthy kernels represent {h_pct:.1f}% of the area, which meets international trading requirements. "
                    f"However, ensure it is dried below 13.5% moisture and packed in hermetic bags for transit."
                )
            else:
                return (
                    f"No, this batch is **not recommended for export** due to its low quality ranking of **{grade}** "
                    f"({d_pct:.1f}% disease, {m_pct:.1f}% missing). "
                    f"It should be diverted for domestic feed mills or local industrial starch processing immediately."
                )
                
        # Default fallback
        else:
            return (
                f"Based on the analysis, this corn is **{grade}** with a quality score of **{context.get('quality_score', 0.0):.1f}/100**.\n"
                f"- Healthy: {h_pct:.1f}%\n"
                f"- Disease: {d_pct:.1f}%\n"
                f"- Missing: {m_pct:.1f}%\n"
                f"- Predicted shelf life: {shelf_life:.1f} days ({risk}) under {st_type}.\n\n"
                f"Please let me know if you want to know about 'Why is this corn {grade}?', 'What caused defects?', 'How long can it be stored?', or 'Is it suitable for export?'."
            )
