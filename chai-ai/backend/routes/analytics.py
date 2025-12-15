"""Analytics routes"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, distinct
from pydantic import BaseModel
from database import get_db
from models import Analytics, User, ChatMessage
from datetime import datetime, timedelta

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


class SessionData(BaseModel):
    session_id: str
    character_id: int
    session_duration: float
    message_count: int


class AnalyticsStats(BaseModel):
    total_users: int
    daily_active_users: int
    total_messages: int
    total_sessions: int
    avg_session_duration: float
    avg_messages_per_session: float


@router.post("/session")
def track_session(session_data: SessionData, db: Session = Depends(get_db)):
    """Track a chat session"""
    analytics = Analytics(
        session_id=session_data.session_id,
        character_id=session_data.character_id,
        session_duration=session_data.session_duration,
        message_count=session_data.message_count
    )
    
    db.add(analytics)
    db.commit()
    
    return {"message": "Session tracked successfully"}


@router.get("/stats", response_model=AnalyticsStats)
def get_stats(db: Session = Depends(get_db)):
    """Get basic analytics statistics"""
    # Total users
    total_users = db.query(func.count(User.id)).scalar()
    
    # Daily active users (users who had sessions in the last 24 hours)
    yesterday = datetime.utcnow() - timedelta(days=1)
    daily_active = db.query(
        func.count(distinct(Analytics.session_id))
    ).filter(Analytics.created_at >= yesterday).scalar()
    
    # Total messages
    total_messages = db.query(func.count(ChatMessage.id)).scalar()
    
    # Total sessions
    total_sessions = db.query(func.count(Analytics.id)).scalar()
    
    # Average session duration
    avg_duration = db.query(func.avg(Analytics.session_duration)).scalar() or 0
    
    # Average messages per session
    avg_messages = db.query(func.avg(Analytics.message_count)).scalar() or 0
    
    return AnalyticsStats(
        total_users=total_users or 0,
        daily_active_users=daily_active or 0,
        total_messages=total_messages or 0,
        total_sessions=total_sessions or 0,
        avg_session_duration=float(avg_duration),
        avg_messages_per_session=float(avg_messages)
    )
