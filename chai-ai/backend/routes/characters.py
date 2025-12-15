"""Character management routes"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from database import get_db
from models import Character, User
from datetime import datetime

router = APIRouter(prefix="/api/characters", tags=["characters"])


class CharacterCreate(BaseModel):
    name: str
    personality: str
    backstory: str
    image_url: Optional[str] = None
    is_public: bool = False


class CharacterUpdate(BaseModel):
    name: Optional[str] = None
    personality: Optional[str] = None
    backstory: Optional[str] = None
    image_url: Optional[str] = None
    is_public: Optional[bool] = None


class CharacterResponse(BaseModel):
    id: int
    name: str
    personality: str
    backstory: str
    image_url: Optional[str]
    is_public: bool
    creator_id: int
    created_at: datetime
    
    class Config:
        from_attributes = True


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
            user = db.query(User).filter(User.session_id == session_id).first()
    return user


@router.post("/", response_model=CharacterResponse)
def create_character(
    character: CharacterCreate,
    session_id: str = "default_session",
    db: Session = Depends(get_db)
):
    """Create a new character"""
    user = get_or_create_user(session_id, db)
    
    db_character = Character(
        name=character.name,
        personality=character.personality,
        backstory=character.backstory,
        image_url=character.image_url,
        is_public=character.is_public,
        creator_id=user.id
    )
    
    db.add(db_character)
    db.commit()
    db.refresh(db_character)
    
    return db_character


@router.get("/", response_model=List[CharacterResponse])
def list_user_characters(
    session_id: str = "default_session",
    db: Session = Depends(get_db)
):
    """List all characters created by the user"""
    user = get_or_create_user(session_id, db)
    characters = db.query(Character).filter(Character.creator_id == user.id).all()
    return characters


@router.get("/community", response_model=List[CharacterResponse])
def list_community_characters(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List all public characters in the community"""
    characters = db.query(Character).filter(
        Character.is_public == True
    ).offset(skip).limit(limit).all()
    return characters


@router.get("/{character_id}", response_model=CharacterResponse)
def get_character(character_id: int, db: Session = Depends(get_db)):
    """Get a specific character by ID"""
    character = db.query(Character).filter(Character.id == character_id).first()
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    return character


@router.put("/{character_id}", response_model=CharacterResponse)
def update_character(
    character_id: int,
    character_update: CharacterUpdate,
    session_id: str = "default_session",
    db: Session = Depends(get_db)
):
    """Update a character"""
    user = get_or_create_user(session_id, db)
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    if character.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to update this character")
    
    # Update fields
    update_data = character_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(character, field, value)
    
    character.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(character)
    
    return character


@router.delete("/{character_id}")
def delete_character(
    character_id: int,
    session_id: str = "default_session",
    db: Session = Depends(get_db)
):
    """Delete a character"""
    user = get_or_create_user(session_id, db)
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    if character.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this character")
    
    db.delete(character)
    db.commit()
    
    return {"message": "Character deleted successfully"}


@router.post("/{character_id}/share")
def share_character(
    character_id: int,
    session_id: str = "default_session",
    db: Session = Depends(get_db)
):
    """Share a character to the community"""
    user = get_or_create_user(session_id, db)
    character = db.query(Character).filter(Character.id == character_id).first()
    
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    if character.creator_id != user.id:
        raise HTTPException(status_code=403, detail="Not authorized to share this character")
    
    character.is_public = True
    db.commit()
    
    return {"message": "Character shared to community successfully"}
