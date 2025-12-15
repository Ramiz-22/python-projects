import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import MessageBubble from '../components/MessageBubble';
import { characterAPI } from '../services/api';
import websocketService from '../services/websocket';
import './Chat.css';

const Chat = () => {
  const { characterId } = useParams();
  const navigate = useNavigate();
  const [character, setCharacter] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    loadCharacter();
    connectWebSocket();

    return () => {
      websocketService.disconnect();
    };
  }, [characterId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadCharacter = async () => {
    try {
      const response = await characterAPI.getById(characterId);
      setCharacter(response.data);
    } catch (error) {
      console.error('Failed to load character:', error);
      navigate('/');
    }
  };

  const handleWebSocketConnect = () => {
    setIsConnected(true);
  };

  const connectWebSocket = () => {
    setIsConnected(false); // Start as false
    websocketService.connect(
      characterId,
      handleWebSocketMessage,
      handleWebSocketError,
      handleWebSocketConnect
    );
  };

  const handleWebSocketMessage = (data) => {
    if (data.type === 'message') {
      // Add bot response only (user message already added optimistically)
      setMessages(prev => [
        ...prev,
        {
          text: data.bot_response,
          isUser: false,
          timestamp: data.timestamp
        }
      ]);
      setIsTyping(false);
    } else if (data.type === 'typing') {
      setIsTyping(data.is_typing);
    } else if (data.type === 'error') {
      alert('Error: ' + data.message);
      setIsTyping(false);
      // Remove failed message? Or show error state? For MVP just alert.
    } else if (data.type === 'system') {
      // Optional: Show system messages as bot messages or toast
      console.log('System message:', data.message);
    }
  };

  const handleWebSocketError = (error) => {
    console.error('WebSocket error:', error);
    setIsConnected(false);
  };

  const handleSendMessage = (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim() || !isConnected) return;

    // 1. Optimistic Update
    const userMsg = {
      text: inputMessage,
      isUser: true,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, userMsg]);
    
    // 2. Send to Server
    websocketService.sendMessage(inputMessage);
    
    // 3. Reset Input
    setInputMessage('');
    setIsTyping(true);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  if (!character) {
    return (
      <div className="chat-loading">
        <div className="loader"></div>
        <p>Loading character...</p>
      </div>
    );
  }

  return (
    <div className="chat">
      <div className="chat__header glass-card">
        <button className="btn-back" onClick={() => navigate('/')}>
          ←
        </button>
        <div className="chat__character-info">
          <div className="chat__character-avatar">
            {character.image_url ? (
              <img src={character.image_url} alt={character.name} />
            ) : (
              <div className="avatar-placeholder">
                {character.name.charAt(0).toUpperCase()}
              </div>
            )}
          </div>
          <div className="chat__character-details">
            <h2 className="gradient-text">{character.name}</h2>
            <p>{character.personality}</p>
          </div>
        </div>
        <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
          {isConnected ? '● Online' : '○ Offline'}
        </div>
      </div>

      <div className="chat__messages">
        {messages.length === 0 && (
          <div className="chat__welcome">
            <h3>Start chatting with {character.name}!</h3>
            <p>{character.backstory}</p>
          </div>
        )}
        
        {messages.map((msg, index) => (
          <MessageBubble
            key={index}
            message={msg.text}
            isUser={msg.isUser}
            timestamp={msg.timestamp}
          />
        ))}
        
        {isTyping && (
          <div className="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      <form className="chat__input-area" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          placeholder="Type your message..."
          disabled={!isConnected}
        />
        <button
          type="submit"
          className="btn btn-primary"
          disabled={!inputMessage.trim() || !isConnected}
        >
          Send
        </button>
      </form>
    </div>
  );
};

export default Chat;
