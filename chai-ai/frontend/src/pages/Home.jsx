import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import CharacterCard from '../components/CharacterCard';
import { characterAPI } from '../services/api';
import './Home.css';

const Home = () => {
  const navigate = useNavigate();
  const [myCharacters, setMyCharacters] = useState([]);
  const [communityCharacters, setCommunityCharacters] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadCharacters();
  }, []);

  const loadCharacters = async () => {
    try {
      const [myChars, communityChars] = await Promise.all([
        characterAPI.getAll(),
        characterAPI.getCommunity(0, 6)
      ]);
      
      setMyCharacters(myChars.data);
      setCommunityCharacters(communityChars.data);
    } catch (error) {
      console.error('Failed to load characters:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCharacterClick = (character) => {
    navigate(`/chat/${character.id}`);
  };

  const handleCreateCharacter = () => {
    navigate('/create');
  };

  const handleBrowseCommunity = () => {
    navigate('/community');
  };

  if (loading) {
    return (
      <div className="home-loading">
        <div className="loader"></div>
        <p>Loading characters...</p>
      </div>
    );
  }

  return (
    <div className="home">
      <section className="hero">
        <div className="container">
          <h1 className="hero__title fade-in">
            Welcome to <span className="gradient-text">Chai AI</span>
          </h1>
          <p className="hero__subtitle fade-in">
            Create and chat with custom AI characters for entertainment and social engagement
          </p>
          <div className="hero__actions fade-in">
            <button className="btn btn-primary" onClick={handleCreateCharacter}>
              Create Character
            </button>
            <button className="btn btn-secondary" onClick={handleBrowseCommunity}>
              Browse Community
            </button>
          </div>
        </div>
      </section>

      {myCharacters.length > 0 && (
        <section className="section">
          <div className="container">
            <h2 className="section__title">My Characters</h2>
            <div className="character-grid">
              {myCharacters.map((character) => (
                <CharacterCard
                  key={character.id}
                  character={character}
                  onClick={handleCharacterClick}
                />
              ))}
            </div>
          </div>
        </section>
      )}

      <section className="section">
        <div className="container">
          <h2 className="section__title">Featured Characters</h2>
          {communityCharacters.length > 0 ? (
            <>
              <div className="character-grid">
                {communityCharacters.map((character) => (
                  <CharacterCard
                    key={character.id}
                    character={character}
                    onClick={handleCharacterClick}
                  />
                ))}
              </div>
              <div className="section__footer">
                <button className="btn btn-secondary" onClick={handleBrowseCommunity}>
                  View All Characters
                </button>
              </div>
            </>
          ) : (
            <p className="empty-state">No community characters yet. Be the first to create one!</p>
          )}
        </div>
      </section>
    </div>
  );
};

export default Home;
