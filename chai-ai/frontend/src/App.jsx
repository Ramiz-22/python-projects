import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import CreateCharacter from './pages/CreateCharacter';
import Chat from './pages/Chat';
import Community from './pages/Community';
import './App.css';

function App() {
  return (
    <Router>
      <div className="app">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/create" element={<CreateCharacter />} />
          <Route path="/chat/:characterId" element={<Chat />} />
          <Route path="/community" element={<Community />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
