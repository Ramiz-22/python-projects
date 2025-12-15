"""WebSocket chat routes"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session
from database import get_db, SessionLocal
from models import Character, ChatMessage, User
from services.llm_service import llm_service
from services.content_filter import content_filter
import json
from typing import Dict, List

router = APIRouter()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, character_id: int):
        await websocket.accept()
        if character_id not in self.active_connections:
            self.active_connections[character_id] = []
        self.active_connections[character_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, character_id: int):
        if character_id in self.active_connections:
            self.active_connections[character_id].remove(websocket)
            if not self.active_connections[character_id]:
                del self.active_connections[character_id]
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)


manager = ConnectionManager()


def get_or_create_user(session_id: str, db: Session) -> User:
    """Get existing user or create anonymous user"""
    user = db.query(User).filter(User.session_id == session_id).first()
    if not user:
        user = User(
            username=f"user_{session_id}",
            session_id=session_id
        )
        db.add(user)
        try:
            db.commit()
            db.refresh(user)
        except Exception:
            db.rollback()
            # Retry fetch if race condition or error
            user = db.query(User).filter(User.session_id == session_id).first()
    return user


@router.websocket("/ws/chat/{character_id}")
async def chat_endpoint(websocket: WebSocket, character_id: int):
    await manager.connect(websocket, character_id)
    db = SessionLocal()
    
    try:
        # Get character from database
        character = db.query(Character).filter(Character.id == character_id).first()
        if not character:
            await websocket.send_json({
                "type": "error",
                "message": "Character not found"
            })
            await websocket.close()
            return
        
        # Send welcome message
        await websocket.send_json({
            "type": "system",
            "message": f"Connected to {character.name}. Start chatting!"
        })
        
        # Get conversation history for context
        conversation_history = []
        
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            session_id = message_data.get("session_id", "default_session")
            
            if not user_message:
                continue
            
            # Filter user message
            if not content_filter.is_appropriate(user_message):
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Message contains inappropriate content"
                }, websocket)
                continue
            
            # Get or create user
            user = get_or_create_user(session_id, db)
            
            # Send typing indicator
            await manager.send_personal_message({
                "type": "typing",
                "is_typing": True
            }, websocket)
            
            try:
                # Generate response using LLM
                bot_response = llm_service.generate_response(
                    user_message=user_message,
                    character_personality=character.personality,
                    character_backstory=character.backstory,
                    conversation_history=conversation_history
                )
                
                # Filter bot response
                bot_response = content_filter.filter_text(bot_response)
                
                # Save to database
                chat_message = ChatMessage(
                    character_id=character_id,
                    user_id=user.id,
                    user_message=user_message,
                    bot_response=bot_response
                )
                db.add(chat_message)
                db.commit()
                
                # Update conversation history
                conversation_history.append({
                    "user": user_message,
                    "bot": bot_response
                })
                
                # Keep only last 10 exchanges in memory
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
                
                # Send response
                await manager.send_personal_message({
                    "type": "message",
                    "user_message": user_message,
                    "bot_response": bot_response,
                    "timestamp": chat_message.timestamp.isoformat()
                }, websocket)
                
            except Exception as e:
                print(f"Error generating response: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Sorry, I had trouble generating a response. Please try again."
                }, websocket)
            
            finally:
                # Stop typing indicator
                await manager.send_personal_message({
                    "type": "typing",
                    "is_typing": False
                }, websocket)
    
    except WebSocketDisconnect:
        manager.disconnect(websocket, character_id)
        print(f"Client disconnected from character {character_id}")
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, character_id)
    
    finally:
        db.close()
