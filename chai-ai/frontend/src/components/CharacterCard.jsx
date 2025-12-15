import React from 'react';
import './CharacterCard.css';

const CharacterCard = ({ character, onClick }) => {
  const handleClick = () => {
    if (onClick) {
      onClick(character);
    }
  };

  return (
    <div className="character-card glass-card fade-in" onClick={handleClick}>
      <div className="character-card__image">
        {character.image_url ? (
          <img src={character.image_url} alt={character.name} />
        ) : (
          <div className="character-card__placeholder">
            {character.name.charAt(0).toUpperCase()}
          </div>
        )}
      </div>
      
      <div className="character-card__content">
        <h3 className="character-card__name gradient-text">{character.name}</h3>
        <p className="character-card__personality">
          {character.personality}
        </p>
        <p className="character-card__backstory">
          {character.backstory.length > 100
            ? character.backstory.substring(0, 100) + '...'
            : character.backstory}
        </p>
      </div>
      
      <div className="character-card__footer">
        <button className="btn btn-primary btn-sm">
          Chat Now
        </button>
      </div>
    </div>
  );
};

export default CharacterCard;
