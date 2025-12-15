import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CharacterCard from '../components/CharacterCard';
import { characterAPI } from '../services/api';
import './Community.css';

const Community = () => {
  const navigate = useNavigate();
  const [characters, setCharacters] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCommunityCharacters();
  }, []);

  const loadCommunityCharacters = async () => {
    try {
      const response = await characterAPI.getCommunity(0, 100);
      setCharacters(response.data);
    } catch (error) {
      console.error('Failed to load community characters:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCharacterClick = (character) => {
    navigate(`/chat/${character.id}`);
  };

  if (loading) {
    return (
      <div className="community-loading">
        <div className="loader"></div>
        <p>Loading community characters...</p>
      </div>
    );
  }

  return (
    <div className="community">
      <div className="container">
        <div className="community__header">
          <button className="btn-back" onClick={() => navigate('/')}>
            ← Back to Home
          </button>
          <h1 className="gradient-text">Community Characters</h1>
          <p className="subtitle">Discover amazing characters created by our community</p>
        </div>

        {characters.length > 0 ? (
          <div className="character-grid">
            {characters.map((character) => (
              <CharacterCard
                key={character.id}
                character={character}
                onClick={handleCharacterClick}
              />
            ))}
          </div>
        ) : (
          <div className="empty-state glass-card">
            <h3>No Community Characters Yet</h3>
            <p>Be the first to create and share a character!</p>
            <button className="btn btn-primary" onClick={() => navigate('/create')}>
              Create Character
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Community;
