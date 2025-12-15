import React from 'react';
import './MessageBubble.css';

const MessageBubble = ({ message, isUser, timestamp }) => {
  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <div className={`message-bubble ${isUser ? 'message-bubble--user' : 'message-bubble--bot'} slide-in`}>
      <div className="message-bubble__content">
        <p className="message-bubble__text">{message}</p>
        {timestamp && (
          <span className="message-bubble__timestamp">{formattedTime}</span>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
