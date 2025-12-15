import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { characterAPI } from '../services/api';
import './CreateCharacter.css';

const CreateCharacter = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    personality: '',
    backstory: '',
    image_url: '',
    is_public: false,
  });
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await characterAPI.create(formData);
      navigate(`/chat/${response.data.id}`);
    } catch (error) {
      console.error('Failed to create character:', error);
      alert('Failed to create character. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="create-character">
      <div className="container">
        <div className="create-character__header">
          <button className="btn-back" onClick={() => navigate('/')}>
            ← Back
          </button>
          <h1 className="gradient-text">Create Your Character</h1>
          <p className="subtitle">Bring your AI character to life</p>
        </div>

        <form className="character-form glass-card" onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Character Name *</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              required
              placeholder="e.g., Alex the Wise"
            />
          </div>

          <div className="form-group">
            <label htmlFor="personality">Personality *</label>
            <input
              type="text"
              id="personality"
              name="personality"
              value={formData.personality}
              onChange={handleChange}
              required
              placeholder="e.g., Friendly, wise, humorous"
            />
            <span className="form-hint">Describe their key traits</span>
          </div>

          <div className="form-group">
            <label htmlFor="backstory">Backstory *</label>
            <textarea
              id="backstory"
              name="backstory"
              value={formData.backstory}
              onChange={handleChange}
              required
              rows="5"
              placeholder="Tell us about your character's background, interests, and what makes them unique..."
            />
            <span className="form-hint">This helps the AI understand how your character should respond</span>
          </div>

          <div className="form-group">
            <label htmlFor="image_url">Image URL (Optional)</label>
            <input
              type="url"
              id="image_url"
              name="image_url"
              value={formData.image_url}
              onChange={handleChange}
              placeholder="https://example.com/image.jpg"
            />
          </div>

          <div className="form-group form-checkbox">
            <label>
              <input
                type="checkbox"
                name="is_public"
                checked={formData.is_public}
                onChange={handleChange}
              />
              <span>Share with community</span>
            </label>
            <span className="form-hint">Allow others to chat with your character</span>
          </div>

          <div className="form-actions">
            <button type="button" className="btn btn-secondary" onClick={() => navigate('/')}>
              Cancel
            </button>
            <button type="submit" className="btn btn-primary" disabled={loading}>
              {loading ? 'Creating...' : 'Create & Chat'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default CreateCharacter;
