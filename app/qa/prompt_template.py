"""
Prompt templates for the RAG system.
Extracted from inline definitions to match the chatbot gold-standard architecture.
Each template uses {context}, {history}, and {question} placeholders.
"""

from langchain_core.prompts import ChatPromptTemplate
from app.config import Config




# ---------------------------------------------------------------------------
# 3. QUERY REPHRASE PROMPT  (Context-aware query rewriting)
# ---------------------------------------------------------------------------
# REPHRASE_TEMPLATE is now managed via Config.REPHRASE_TEMPLATE and stored in SQLite.


# ---------------------------------------------------------------------------
# Helper: get the right prompt for a given mode
# ---------------------------------------------------------------------------
def get_chat_prompt(deep_thinking: bool = False) -> ChatPromptTemplate:
    """Return the appropriate ChatPromptTemplate based on mode."""
    template = Config.DOCUMENT_TEMPLATE if deep_thinking else Config.CHAT_TEMPLATE
    return ChatPromptTemplate.from_template(template)


def get_rephrase_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(Config.REPHRASE_TEMPLATE)
