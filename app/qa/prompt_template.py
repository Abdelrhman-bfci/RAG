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
REPHRASE_TEMPLATE = """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.
The standalone question must be fully self-contained and understood without the chat history. Do not answer the question, just rephrase it.

Chat History:
{history}

Follow Up Input:
{question}

Standalone Question:"""


# ---------------------------------------------------------------------------
# Helper: get the right prompt for a given mode
# ---------------------------------------------------------------------------
def get_chat_prompt(deep_thinking: bool = False) -> ChatPromptTemplate:
    """Return the appropriate ChatPromptTemplate based on mode."""
    template = Config.DOCUMENT_TEMPLATE if deep_thinking else Config.CHAT_TEMPLATE
    return ChatPromptTemplate.from_template(template)


def get_rephrase_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template(REPHRASE_TEMPLATE)
