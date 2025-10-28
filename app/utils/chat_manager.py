
# app/backend/chat_manager.py

import uuid
from datetime import datetime
import random

class ChatSession:
    def __init__(self, session_id=None):
        """Initialize a chat session"""
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_history = []
        self.start_time = datetime.now()
        self.last_active = datetime.now()
    
    def add_message(self, role, content, emotion=None):
        """Add a message to the conversation history"""
        message = {
            'role': role,  # 'user' or 'bot'
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'emotion': emotion
        }
        self.conversation_history.append(message)
        self.last_active = datetime.now()
        return message
    
    def get_history(self):
        """Get the conversation history"""
        return self.conversation_history


class ChatManager:
    def __init__(self):
        """Initialize the chat manager"""
        self.active_sessions = {}
        self.goodbye_responses = [
            "It was wonderful chatting with you! Take care and remember I'm here if you need a bit of encouragement in the future.",
            "Thank you for chatting with me today. I hope our conversation brought some positivity to your day. Take care!",
            "I've enjoyed our conversation. Remember that you're doing great, and I'm here whenever you need a boost. Goodbye for now!",
            "It's been a pleasure talking with you. Stay positive and take care of yourself. Until next time!",
            "Wishing you all the best! Remember your strength and resilience. I'll be here if you need encouragement in the future."
        ]
    
    def get_or_create_session(self, session_id):
        """Get an existing session or create a new one"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = ChatSession(session_id)
        return self.active_sessions[session_id]
    
    def end_session(self, session_id):
        """End a chat session"""
        if session_id in self.active_sessions:
            # the session can be stored for later analysis
            # For now, we'll just generate a goodbye message
            goodbye_msg = random.choice(self.goodbye_responses)
            self.active_sessions[session_id].add_message('bot', goodbye_msg, 'neutral')
            return goodbye_msg
        return None
    
    def is_goodbye_message(self, message):
        """Check if a message is a goodbye"""
        # Convert to lowercase for easier matching
        text_lower = message.lower()
        
        # Common goodbye phrases
        goodbye_phrases = [
             'exit',  'end chat'
        ]
        
        # Check if any goodbye phrase is in the text
        for phrase in goodbye_phrases:
            if phrase in text_lower:
                return True
                
        # Check for shorter versions with whole word matching
        short_goodbyes = ['bye', 'cya', 'gtg']
        for word in text_lower.split():
            if word in short_goodbyes:
                return True
                
        return False