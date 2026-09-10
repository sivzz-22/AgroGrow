"""
AgroGrow AI Assistant — Legacy Compatibility Module.
This module is retained for backward compatibility only.
The active chatbot implementation has moved to:
    AgroGrow/assistant/chatbot.py  (Gemini AI + rule-based CornBot)

Do NOT add new code here. Use chatbot.py instead.
"""

# Re-export the new chatbot for any code that still imports from this module
from AgroGrow.assistant.chatbot import CornChatbot as CornAssistant

__all__ = ["CornAssistant"]
