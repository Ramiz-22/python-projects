import { getSessionId } from './api';

class WebSocketService {
  constructor() {
    this.ws = null;
    this.characterId = null;
    this.messageHandlers = [];
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect(characterId, onMessage, onError, onConnect) {
    this.characterId = characterId;
    
    // Update callbacks if provided, otherwise keep existing (for reconnection)
    if (onError) this.onError = onError;
    if (onConnect) this.onConnect = onConnect;
    
    // Add message handler if provided
    if (onMessage) {
      // Clear previous to avoid duplicates on strict mode double-invoke
      this.messageHandlers = [onMessage]; 
    }
    
    // Use the same host/port as frontend to leverage Vite proxy
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/chat/${characterId}`;
    console.log('Connecting to WebSocket:', wsUrl);
    
    try {
      this.ws = new WebSocket(wsUrl);
      
      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        if (this.onConnect) this.onConnect();
      };
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.messageHandlers.forEach(handler => handler(data));
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (this.onError) this.onError(error);
      };
      
      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.attemptReconnect();
      };
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      if (this.onError) this.onError(error);
    }
  }

  attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      console.log(`Attempting to reconnect... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        if (this.characterId) {
          // Reconnect using stored callbacks
          this.connect(this.characterId);
        }
      }, 2000 * this.reconnectAttempts);
    }
  }

  addMessageHandler(handler) {
    this.messageHandlers.push(handler);
  }

  removeMessageHandler(handler) {
    this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
  }

  sendMessage(message) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      const data = {
        message,
        session_id: getSessionId(),
      };
      this.ws.send(JSON.stringify(data));
    } else {
      console.error('WebSocket is not connected');
      this.onError(new Error('WebSocket is not connected'));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.messageHandlers = [];
    this.characterId = null;
  }

  isConnected() {
    return this.ws && this.ws.readyState === WebSocket.OPEN;
  }
}

export default new WebSocketService();
