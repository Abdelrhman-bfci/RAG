import time
import uuid
from typing import List, Dict, Optional
from collections import defaultdict

# In-memory storage
_sessions = {}  # {session_id: {"created_at": float, "last_updated": float, "messages": []}}
MAX_HISTORY_MESSAGES = 10  # Configurable limit for conversation history

def create_session() -> str:
    """Create a new chat session and return its ID."""
    session_id = str(uuid.uuid4())
    timestamp = time.time()
    
    _sessions[session_id] = {
        "created_at": timestamp,
        "last_updated": timestamp,
        "messages": []
    }
    
    return session_id

def get_session_history(session_id: str, limit: int = MAX_HISTORY_MESSAGES) -> List[Dict[str, str]]:
    """
    Get conversation history for a session.
    Returns list of messages in format: [{"role": "user", "content": "..."}, ...]
    """
    if session_id not in _sessions:
        return []
    
    messages = _sessions[session_id]["messages"]
    # Return the most recent messages, up to the limit
    return messages[-limit:] if len(messages) > limit else messages

def add_message(session_id: str, role: str, content: str):
    """Add a message to the session history."""
    if session_id not in _sessions:
        # Auto-create session if it doesn't exist
        create_session_with_id(session_id)
    
    timestamp = time.time()
    
    _sessions[session_id]["messages"].append({
        "role": role,
        "content": content,
        "timestamp": timestamp
    })
    
    _sessions[session_id]["last_updated"] = timestamp

def create_session_with_id(session_id: str):
    """Create a session with a specific ID (for auto-creation)."""
    timestamp = time.time()
    _sessions[session_id] = {
        "created_at": timestamp,
        "last_updated": timestamp,
        "messages": []
    }

def delete_session(session_id: str):
    """Delete a session and all its messages."""
    if session_id in _sessions:
        del _sessions[session_id]

def clear_all_sessions():
    """Clear all active sessions (wipes chat history)."""
    global _sessions
    _sessions.clear()
    print("All chat sessions cleared.")

def get_all_sessions() -> List[Dict]:
    """Get all chat sessions with metadata."""
    sessions = []
    for session_id, data in _sessions.items():
        sessions.append({
            "id": session_id,
            "created_at": data["created_at"],
            "last_updated": data["last_updated"],
            "message_count": len(data["messages"])
        })
    
    # Sort by last_updated, most recent first
    sessions.sort(key=lambda x: x["last_updated"], reverse=True)
    return sessions

def session_exists(session_id: str) -> bool:
    """Check if a session exists."""
    return session_id in _sessions

def format_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """
    Format conversation history for inclusion in the LLM prompt.
    """
    if not history:
        return ""
    
    formatted = "Previous Conversation:\n"
    for msg in history:
        if not isinstance(msg, dict):
            # Skip if msg is somehow a string or other type
            continue
        role_raw = msg.get("role", "user")
        role = "User" if role_raw == "user" else "Assistant"
        content = msg.get("content", "")
        if not content:
            continue
        formatted += f"{role}: {content}\n\n"
    
    return formatted
