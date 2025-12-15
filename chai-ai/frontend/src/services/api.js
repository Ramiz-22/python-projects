import axios from 'axios';

// Create axios instance
const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Get or create session ID
const getSessionId = () => {
  let sessionId = localStorage.getItem('sessionId');
  if (!sessionId) {
    sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('sessionId', sessionId);
  }
  return sessionId;
};

// Character API
export const characterAPI = {
  create: (characterData) => 
    api.post('/characters/', characterData, {
      params: { session_id: getSessionId() }
    }),
  
  getAll: () => 
    api.get('/characters/', {
      params: { session_id: getSessionId() }
    }),
  
  getById: (id) => 
    api.get(`/characters/${id}`),
  
  update: (id, characterData) => 
    api.put(`/characters/${id}`, characterData, {
      params: { session_id: getSessionId() }
    }),
  
  delete: (id) => 
    api.delete(`/characters/${id}`, {
      params: { session_id: getSessionId() }
    }),
  
  share: (id) => 
    api.post(`/characters/${id}/share`, {}, {
      params: { session_id: getSessionId() }
    }),
  
  getCommunity: (skip = 0, limit = 50) => 
    api.get('/characters/community', {
      params: { skip, limit }
    }),
};

// Analytics API
export const analyticsAPI = {
  trackSession: (sessionData) => 
    api.post('/analytics/session', sessionData),
  
  getStats: () => 
    api.get('/analytics/stats'),
};

export { getSessionId };
export default api;
