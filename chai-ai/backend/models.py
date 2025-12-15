from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True)
    session_id = Column(String(255), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    characters = relationship("Character", back_populates="creator")
    messages = relationship("ChatMessage", back_populates="user")
    analytics = relationship("Analytics", back_populates="user")


class Character(Base):
    __tablename__ = "characters"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    personality = Column(Text, nullable=False)
    backstory = Column(Text, nullable=False)
    image_url = Column(String(500), nullable=True)
    is_public = Column(Boolean, default=False)
    creator_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    creator = relationship("User", back_populates="characters")
    messages = relationship("ChatMessage", back_populates="character")


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    character_id = Column(Integer, ForeignKey("characters.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    character = relationship("Character", back_populates="messages")
    user = relationship("User", back_populates="messages")


class Analytics(Base):
    __tablename__ = "analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    character_id = Column(Integer, nullable=True)
    session_duration = Column(Float, default=0.0)  # in seconds
    message_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="analytics")
