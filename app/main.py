# app/main.py

import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import logging
import uvicorn
import time
from typing import Dict, List
import asyncio

# Import our backend components
from models.emotion_detector import EmotionDetector
from models.response_generator import ResponseGenerator
from utils.chat_manager import ChatManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="EmotiveChat")

# Set up static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Helper function to clean response text
def clean_response_text(text):
    """Clean up any formatting placeholders in the response text"""
    cleaned = text.replace("_comma_", ",")
    cleaned = cleaned.replace("no_comma_", "")
    # Add any other replacements you notice
    return cleaned

# Initialize our backend components
try:
    # Define base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    emotion_model_path = os.path.join(base_dir, 'models', 'emotion_classifier_model')
    response_model_path = os.path.join(base_dir, 'models', 'response_model')
    
    # Ensure paths exist
    if not os.path.exists(emotion_model_path):
        logger.error(f"Emotion model path not found: {emotion_model_path}")
        raise FileNotFoundError(f"Emotion model path not found: {emotion_model_path}")
    
    if not os.path.exists(response_model_path):
        logger.error(f"Response model path not found: {response_model_path}")
        raise FileNotFoundError(f"Response model path not found: {response_model_path}")
    #initialize models   
    emotion_detector = EmotionDetector(
        use_pretrained=False, 
        model_path='models/emotion_classifier_model'
    )
    response_generator = ResponseGenerator(
        emotion_detector,
        model_path='models/response_model'
    )
except Exception as e:
    logger.error(f"Error initializing models: {e}")
    #  fallback behaviour here or exit gracefully
    raise

# Initialize chat manager
chat_manager = ChatManager()

# Store active websocket connections
active_connections: Dict[str, WebSocket] = {}

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def get_root():
    return FileResponse("static/index.html")

# WebSocket connection
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    ping_interval = 30  # seconds
    last_ping_time = time.time()
    
    # Store the connection
    active_connections[session_id] = websocket
    
    # Get or create session
    session = chat_manager.get_or_create_session(session_id)
    
    # Send welcome message
    welcome_msg = "Hi there! I'm EmpathBot, here to brighten your day"
    session.add_message('bot', welcome_msg, 'neutral')
    
    await websocket.send_json({
        'type': 'message',
        'content': welcome_msg,
        'role': 'bot',
        'emotion': 'neutral'
    })

    
    
    try:
        while True:
            # Check if we need to send a ping
            current_time = time.time()
            if current_time - last_ping_time > ping_interval:
                try:
                    await websocket.send_json({"type": "ping"})
                    last_ping_time = current_time
                except:
                    break
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
  
            if message_data.get('type') == 'message':
                user_message = message_data.get('content', '')
                
                # Add to conversation history
                session.add_message('user', user_message)
                
                # Check if this is a goodbye message
                if chat_manager.is_goodbye_message(user_message):
                    # Send goodbye message
                    goodbye_msg = chat_manager.end_session(session_id)
                    
                    await websocket.send_json({
                        'type': 'message',
                        'content': goodbye_msg,
                        'role': 'bot',
                        'emotion': 'neutral',
                        'is_goodbye': True
                    })
                    
                    # End chat session
                    await websocket.send_json({
                        'type': 'chat_ended'
                    })
                    
                else:
                    # Generate response
                    try:
                        result = response_generator.generate_response(user_message)
                        bot_message = result['response']
                        emotion = result.get('detected_emotion', 'neutral')  # Default to neutral if missing
                        
                        # Clean up any formatting issues
                        bot_message = clean_response_text(bot_message)
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        bot_message = "I'm having trouble processing that right now. Could you try again?"
                        emotion = "neutral"
                    
                    # Add to conversation history
                    session.add_message('bot', bot_message, emotion)
                    
                    # Send response
                    await websocket.send_json({
                        'type': 'message',
                        'content': bot_message,
                        'role': 'bot',
                        'emotion': emotion
                    })
                    
            elif message_data.get('type') == 'feedback':
                # Handle feedback (emoji reactions)
                feedback = message_data.get('feedback')
                message_id = message_data.get('message_id')
                
                # You could store this feedback for model improvement
                logger.info(f"Received feedback: {feedback} for message {message_id}")
                
                await websocket.send_json({
                    'type': 'acknowledgment',
                    'text': 'Thanks for your feedback!'
                })
                
            elif message_data.get('type') == 'end_chat':
                # Explicitly end the chat
                goodbye_msg = chat_manager.end_session(session_id)
                
                await websocket.send_json({
                    'type': 'message',
                    'content': goodbye_msg,
                    'role': 'bot',
                    'emotion': 'neutral',
                    'is_goodbye': True
                })
                
                # End chat session
                await websocket.send_json({
                    'type': 'chat_ended'
                })
                
                break
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"Error in websocket connection: {e}")
        if session_id in active_connections:
            del active_connections[session_id]

# App startup event
@app.on_event("startup")
async def startup_event():
    logger.info("Starting EmotiveChat application")

# App shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down EmotiveChat application")

# Run the application if executed directly
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)