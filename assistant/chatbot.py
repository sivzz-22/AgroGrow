"""
AgroGrow Corn Variety Detection + Chatbot Assistant Module.
Provides a Gemini-powered (or rule-based fallback) chatbot for answering
general and context-specific questions about corn quality, varieties, and storage.
"""

import os
import re
from typing import Dict, Optional
from dotenv import load_dotenv
from AgroGrow.utils.logger import logger

load_dotenv()

# ── Corn Variety Knowledge Base ───────────────────────────────────────────────
VARIETY_INFO = {
    "dent": {
        "full_name": "Commercial Dent Corn (Zea mays indentata)",
        "description": (
            "Dent corn is the most widely grown corn variety globally. Named after the distinctive "
            "dent that forms in the crown of each kernel as it dries. Kernels are typically deep "
            "yellow or white. Used extensively for animal feed, industrial starch, corn syrup, "
            "ethanol production, and processed food products like corn flour and grits."
        ),
        "colour": "Deep yellow or white kernels; the dent creates a characteristic sunken tip.",
        "uses": "Animal feed, starch, ethanol, corn flour, corn syrup, processed foods.",
        "shelf_life_note": "Can store up to 5–10 years when dried below 13% moisture and kept sealed."
    },
    "flint": {
        "full_name": "Indian / Multi-Colored Flint Corn (Zea mays indurata)",
        "description": (
            "Flint corn has a very hard outer shell (endosperm), making it resistant to spoilage "
            "and freezing damage. Indian or ornamental corn belongs to this group and is characterised "
            "by striking multicoloured kernels — ruby red, deep purple, bronze, orange, and cream — "
            "caused by anthocyanin pigmentation. Varieties include Bloody Butcher, Glass Gem, and Calico."
        ),
        "colour": "Ruby red, purple, bronze, orange, cream — anthocyanin pigments create vivid multicolour patterns.",
        "uses": "Human food (polenta, tortillas, pozole), ornamental/decorative, alcohol brewing.",
        "shelf_life_note": "Exceptionally long shelf life due to hard endosperm — 10+ years properly stored."
    },
    "sweet": {
        "full_name": "Sweet Corn (Zea mays saccharata)",
        "description": (
            "Sweet corn has a genetic mutation that delays sugar conversion to starch, resulting in "
            "naturally sweet, tender kernels. Kernels are typically pale cream-yellow, wrinkled when dried, "
            "and very high in simple sugars. It is the corn eaten directly as a vegetable — on the cob, "
            "canned, or frozen."
        ),
        "colour": "Pale cream or light yellow; kernels appear plump and shiny when fresh, wrinkled when dried.",
        "uses": "Fresh vegetable consumption, canned corn, frozen corn, baby corn.",
        "shelf_life_note": "Very short post-harvest life — best consumed within 1–3 days of harvest."
    },
    "popcorn": {
        "full_name": "Popcorn (Zea mays everta)",
        "description": (
            "Popcorn is a unique variety with a very hard, moisture-sealed pericarp (outer shell). "
            "When heated, internal moisture vaporises and expands rapidly, causing the kernel to explode "
            "and invert. Kernels are typically small, hard, pearl-white to yellow, or red."
        ),
        "colour": "Small, round, hard kernels — pearl white, yellow, or red varieties.",
        "uses": "Snack food exclusively (popped corn). Some varieties used in artisan products.",
        "shelf_life_note": "Stores well up to 2 years if kept below 14% moisture and sealed from humidity."
    },
    "blue": {
        "full_name": "Blue / Black Corn (Zea mays — Hopi Blue variety)",
        "description": (
            "Blue corn is an ancient variety originating from the American Southwest and Mexico, "
            "particularly significant in Hopi and Navajo cultures. Its dark blue-purple or grey-black "
            "colour comes from high concentrations of anthocyanins — powerful antioxidants. "
            "Blue corn has higher protein content than yellow dent corn and a nuttier flavour."
        ),
        "colour": "Deep blue-purple to grey-black kernels, uniform dark colouring.",
        "uses": "Blue corn tortillas, chips, atole, ceremonial use, specialty health foods.",
        "shelf_life_note": "Similar to Flint corn — stores well long-term when properly dried and sealed."
    }
}

GENERAL_CORN_FACTS = {
    "disease": (
        "Common corn diseases include: Gray Leaf Spot (Cercospora zeae-maydis), Northern Corn Leaf Blight "
        "(Setosphaeria turcica), Gibberella Ear Rot, Fusarium Ear Rot, Corn Smut (Ustilago maydis), "
        "and Aflatoxin contamination by Aspergillus flavus. Disease-infected corn kernels typically appear "
        "shrunken, discoloured (pink, grey, white-mouldy), or sunken. The AgroGrow system flags these as "
        "red pixels in the segmentation overlay."
    ),
    "storage": (
        "Corn storage best practices: maintain grain moisture below 13.5% for long-term storage; "
        "store at 10-15°C in well-ventilated bins or hermetic bags; inspect regularly for mould, "
        "insects, and heating; avoid mixing new and old grain; use appropriate fungicides or "
        "insecticides only where legally approved. Cold storage (-5°C to 5°C) dramatically extends shelf life."
    ),
    "grading": (
        "Corn grading follows USDA and ISO standards: Grade A means >88% healthy kernels with <4% disease; "
        "Grade B means >75% healthy with <8% disease; Grade C means >55% healthy with <16% disease; "
        "Grade D (Sample Grade) indicates high contamination — not suitable for direct food use."
    )
}


class CornChatbot:
    """
    Corn expert chatbot. Uses Google Gemini API when GEMINI_API_KEY is set,
    otherwise falls back to an enhanced context-aware rule-based system.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
        self.gemini_client = None
        self.gemini_model = None
        self.model_name = "gemini-flash-latest"
        self._init_gemini()

    def _init_gemini(self):
        load_dotenv(override=True)
        env_key = os.getenv("GEMINI_API_KEY", "").strip()
        if env_key:
            self.api_key = env_key

        if self.api_key and not self.gemini_model:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                for candidate in ["gemini-flash-latest", "gemini-3.6-flash", "gemini-2.5-flash-lite"]:
                    try:
                        self.gemini_model = genai.GenerativeModel(candidate)
                        self.model_name = candidate
                        logger.info(f"Gemini AI chatbot initialised successfully with model: {candidate}")
                        break
                    except Exception:
                        continue
            except ImportError:
                logger.warning("google-generativeai not installed. Falling back to rule-based chatbot.")
                self.gemini_model = None
            except Exception as e:
                logger.warning(f"Gemini init failed: {e}. Using rule-based fallback.")
                self.gemini_model = None
        elif not self.api_key:
            logger.info("No GEMINI_API_KEY found. Running enhanced rule-based chatbot.")

    @property
    def is_ai_powered(self) -> bool:
        if self.gemini_model is None:
            self._init_gemini()
        return self.gemini_model is not None

    def chat(self, user_message: str, analysis_context: Optional[Dict] = None) -> str:
        """
        Responds to a user message.

        Args:
            user_message: The question from the user.
            analysis_context: Optional dict with current analysis results
                              (grade, percentages, shelf_life, variety, etc.)

        Returns:
            str: The chatbot response.
        """
        if self.gemini_model is not None:
            return self._gemini_response(user_message, analysis_context)
        else:
            return self._rule_based_response(user_message, analysis_context)

    # Alias for flexibility
    ask = chat

    def _build_system_prompt(self, context: Optional[Dict]) -> str:
        """Builds the Gemini system prompt with injected analysis context."""
        base = (
            "You are CornBot, an expert agronomist assistant specializing in post-harvest "
            "corn quality assessment, grain grading, storage science, and corn varieties. "
            "You are embedded in the AgroGrow platform — an AI-powered corn quality assessment tool.\n\n"
            "You know about these corn varieties in detail:\n"
            "- Dent Corn (most common, yellow/white, used for feed and industrial purposes)\n"
            "- Flint / Indian Corn (hard endosperm, multicoloured anthocyanin pigmentation — ruby, purple, bronze)\n"
            "- Sweet Corn (tender, high sugar, eaten as vegetable)\n"
            "- Popcorn (small hard kernels, unique explosion mechanism)\n"
            "- Blue / Black Corn (high anthocyanins, Hopi origin, nuttier flavour)\n\n"
            "Guidelines:\n"
            "- Give specific, accurate, professional answers.\n"
            "- Do NOT give generic placeholder responses.\n"
            "- Keep answers under 150 words unless a detailed explanation is specifically requested.\n"
            "- When analysis results are available, reference them directly in your answer.\n"
            "- Use markdown bold (**text**) for key terms.\n"
        )

        if context:
            grade = context.get("grade", "N/A")
            quality = context.get("quality_score", 0.0)
            healthy = context.get("healthy_percentage", 0.0)
            disease = context.get("disease_percentage", 0.0)
            missing = context.get("missing_percentage", 0.0)
            shelf_life = context.get("shelf_life_days", 0.0)
            risk = context.get("risk_level", "N/A")
            storage_type = context.get("storage_type", "N/A")
            temp = context.get("temperature_c", "N/A")
            humidity = context.get("humidity_pct", "N/A")
            variety = context.get("variety", "N/A")
            summary = context.get("summary", "")

            base += (
                f"\n\nCURRENT ANALYSIS RESULTS (reference these when relevant):\n"
                f"- Detected Variety: {variety}\n"
                f"- Quality Grade: {grade}\n"
                f"- Quality Score: {quality:.1f}/100\n"
                f"- Healthy Kernels: {healthy:.1f}%\n"
                f"- Diseased Kernels: {disease:.1f}%\n"
                f"- Missing Kernels: {missing:.1f}%\n"
                f"- Estimated Shelf Life: {shelf_life:.0f} days\n"
                f"- Storage Risk: {risk}\n"
                f"- Storage Type: {storage_type} at {temp}°C, {humidity}% RH\n"
                f"- Grade Summary: {summary}\n"
            )

        return base

    def _gemini_response(self, user_message: str, context: Optional[Dict]) -> str:
        """Gets a response from the Gemini API."""
        system_prompt = self._build_system_prompt(context)
        full_prompt = f"{system_prompt}\n\nUser question: {user_message}"
        try:
            response = self.gemini_model.generate_content(full_prompt)
            if response and response.text:
                return response.text.strip()
            return self._rule_based_response(user_message, context)
        except Exception as e:
            logger.error(f"Gemini API error ({getattr(self, 'model_name', 'unknown')}): {e}")
            for fallback in ["gemini-3.6-flash", "gemini-flash-latest"]:
                if fallback != getattr(self, "model_name", ""):
                    try:
                        import google.generativeai as genai
                        self.gemini_model = genai.GenerativeModel(fallback)
                        self.model_name = fallback
                        resp = self.gemini_model.generate_content(full_prompt)
                        if resp and resp.text:
                            return resp.text.strip()
                    except Exception:
                        continue
            return self._rule_based_response(user_message, context)

    def _rule_based_response(self, user_message: str, context: Optional[Dict]) -> str:
        """Enhanced rule-based fallback — context-aware, not canned."""
        q = user_message.lower().strip()

        # ── Variety queries ───────────────────────────────────────────────────
        for key, info in VARIETY_INFO.items():
            if key in q or any(word in q for word in info["full_name"].lower().split()):
                detected = context.get("variety", "") if context else ""
                match_note = ""
                if key in detected.lower():
                    match_note = f"\n\n🌽 **This matches your current analysis** — the uploaded image was identified as {detected}."
                return (
                    f"**{info['full_name']}**\n\n"
                    f"{info['description']}\n\n"
                    f"**Colour:** {info['colour']}\n"
                    f"**Uses:** {info['uses']}\n"
                    f"**Storage:** {info['shelf_life_note']}"
                    f"{match_note}"
                )

        # ── Context-aware responses (need analysis loaded) ────────────────────
        if context:
            grade = context.get("grade", "N/A")
            quality = context.get("quality_score", 0.0)
            healthy = context.get("healthy_percentage", 0.0)
            disease = context.get("disease_percentage", 0.0)
            missing = context.get("missing_percentage", 0.0)
            shelf_life = context.get("shelf_life_days", 0.0)
            risk = context.get("risk_level", "N/A")
            variety = context.get("variety", "Unknown variety")
            temp = context.get("temperature_c", 25.0)
            storage = context.get("storage_type", "Open Air")

            # Why this grade?
            if "why" in q and "grade" in q:
                return (
                    f"This batch is classified as **{grade}** based on its kernel composition:\n\n"
                    f"- **{healthy:.1f}%** healthy kernels\n"
                    f"- **{disease:.1f}%** diseased kernels\n"
                    f"- **{missing:.1f}%** missing sockets\n\n"
                    f"Grade A requires ≥88% healthy and <4% disease. Grade B requires ≥75% healthy. "
                    f"Grade C needs ≥55% healthy. Your corn's quality score is **{quality:.1f}/100**, "
                    f"placing it firmly in the **{grade}** category."
                )

            # Shelf life / storage questions
            if any(w in q for w in ["shelf", "stor", "how long", "expire", "last"]):
                return (
                    f"Under **{storage}** at **{temp}°C**, this batch has an estimated shelf life of "
                    f"**{shelf_life:.0f} days** with a **{risk}** profile.\n\n"
                    f"{GENERAL_CORN_FACTS['storage']}"
                )

            # Disease / defect questions
            if any(w in q for w in ["disease", "rot", "fungal", "defect", "infected", "sick"]):
                if disease < 3.0:
                    return (
                        f"Good news — this batch shows only **{disease:.1f}%** diseased kernels, "
                        f"well within acceptable limits for {grade}. "
                        f"No immediate disease risk detected. Continue standard storage protocols."
                    )
                else:
                    return (
                        f"Disease detected at **{disease:.1f}%** of kernels. "
                        f"{GENERAL_CORN_FACTS['disease']}\n\n"
                        f"For this batch, ensure grain is dried below 13% moisture immediately "
                        f"and consider isolating the batch to prevent spread."
                    )

            # Export suitability
            if any(w in q for w in ["export", "sell", "suitable", "market"]):
                if "A" in grade or "B" in grade:
                    return (
                        f"✅ **Yes**, this batch qualifies for export as **{grade}**. "
                        f"With {healthy:.1f}% healthy kernels and only {disease:.1f}% disease, "
                        f"it meets international trading standards. Ensure moisture is below 13.5% "
                        f"and pack in hermetic bags for transit."
                    )
                else:
                    return (
                        f"❌ **Not recommended for export** — the batch is classified as **{grade}** "
                        f"with {disease:.1f}% disease and {missing:.1f}% missing kernels. "
                        f"Consider diverting to domestic animal feed or industrial processing."
                    )

            # Variety questions in context
            if any(w in q for w in ["variety", "type", "which corn", "what corn", "what kind"]):
                return (
                    f"The uploaded image was identified as **{variety}**.\n\n"
                    + self._variety_detail(variety)
                )

        # ── General corn knowledge ────────────────────────────────────────────
        if any(w in q for w in ["disease", "rot", "fungal", "blight", "smut"]):
            return GENERAL_CORN_FACTS["disease"]

        if any(w in q for w in ["grade", "grading", "quality standard", "usda"]):
            return GENERAL_CORN_FACTS["grading"]

        if any(w in q for w in ["store", "storage", "hermetic", "silo", "moisture"]):
            return GENERAL_CORN_FACTS["storage"]

        if any(w in q for w in ["variety", "varieties", "types of corn", "kinds"]):
            return (
                "AgroGrow can detect and explain **5 corn varieties**:\n\n"
                "1. 🌽 **Dent Corn** — most common worldwide, yellow/white, for feed & industry\n"
                "2. 🎨 **Indian / Flint Corn** — multicoloured (ruby, purple, bronze), hard endosperm\n"
                "3. 🍬 **Sweet Corn** — pale cream, tender, eaten as vegetable\n"
                "4. 🍿 **Popcorn** — small hard kernels, only variety that pops\n"
                "5. 🔵 **Blue / Black Corn** — dark anthocyanin pigmentation, Hopi origin\n\n"
                "Ask me about any specific variety for detailed information!"
            )

        if any(w in q for w in ["hello", "hi", "hey", "how are", "what can you"]):
            return (
                "👋 Hello! I'm **CornBot**, your corn quality expert.\n\n"
                "I can help you with:\n"
                "- 🌽 **Corn variety identification** (Dent, Flint, Sweet, Popcorn, Blue corn)\n"
                "- 🔬 **Disease diagnosis** (what diseases look like, how to manage them)\n"
                "- 📦 **Storage advice** (temperature, moisture, hermetic bags)\n"
                "- 📊 **Quality grading** (Grade A/B/C/D standards)\n"
                "- 🌾 **Current analysis results** (ask about your uploaded image)\n\n"
                "Just ask me anything about corn!"
            )

        # Default — still context-aware
        if context:
            grade = context.get("grade", "N/A")
            variety = context.get("variety", "corn")
            return (
                f"I'm your corn quality expert! The current batch ({variety}) is **{grade}**. "
                f"You can ask me: *'Why is this corn {grade}?'*, *'How long can it be stored?'*, "
                f"*'Is it suitable for export?'*, or *'Tell me about flint corn'*."
            )

        return (
            "I'm **CornBot**, your corn quality expert! Ask me about:\n"
            "- Corn varieties (Dent, Flint, Sweet, Popcorn, Blue corn)\n"
            "- Disease identification and storage best practices\n"
            "- Quality grading standards (Grade A/B/C/D)\n"
            "- Your current analysis results\n\n"
            "What would you like to know?"
        )

    def _variety_detail(self, variety_str: str) -> str:
        """Returns a short description for the detected variety string."""
        v = variety_str.lower()
        if "flint" in v or "indian" in v:
            info = VARIETY_INFO["flint"]
        elif "sweet" in v:
            info = VARIETY_INFO["sweet"]
        elif "popcorn" in v or "pop corn" in v:
            info = VARIETY_INFO["popcorn"]
        elif "blue" in v or "black" in v:
            info = VARIETY_INFO["blue"]
        else:
            info = VARIETY_INFO["dent"]
        return f"{info['description']}\n\n**Colour:** {info['colour']}\n**Uses:** {info['uses']}"
