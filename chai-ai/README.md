# Chai AI Clone 🤖✨

A powerful, modern chatbot application inspired by Chai AI, featuring real-time roleplay capability powered by **DeepSeek** and **Google Gemini**.

## 🚀 Features
- **Intelligent Conversations**: Integrated with DeepSeek-V3 via OpenRouter for high-quality roleplay.
- **Multimodal Support**: Fallback to Google Gemini 1.5 Flash or Pro.
- **Real-Time Chat**: WebSocket-based messaging with optimistic UI updates.
- **Custom Characters**: Create, edit, and chat with unique personas.
- **Modern UI**: Glassmorphism design using React and Vanilla CSS.

## 🛠️ Tech Stack
- **Backend**: FastAPI (Python), SQLAlchemy, WebSockets.
- **Frontend**: React (Vite), CSS3.
- **AI**: DeepSeek API / Google Gemini API.

## 📦 Installation

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration
Create a `.env` file in the `backend/` directory:
```ini
# DeepSeek (Recommended)
AI_PROVIDER=deepseek
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://openrouter.ai/api/v1
DEEPSEEK_MODEL=deepseek/deepseek-chat

# Optional: Google Fallback
GOOGLE_API_KEY=your_google_key
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 4. Run
Start Backend:
```bash
python backend/main.py
```
Start Frontend:
```bash
cd frontend
npm run dev
```
Open [http://localhost:5173](http://localhost:5173).

## 📄 License
MIT
