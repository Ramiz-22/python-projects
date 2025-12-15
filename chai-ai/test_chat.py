"""Test WebSocket chat with Luna the Mystic"""
import asyncio
import websockets
import json

async def test_chat():
    uri = "ws://localhost:8000/ws/chat/1"
    
    print("🔌 Connecting to Luna the Mystic...")
    
    async with websockets.connect(uri) as websocket:
        # Wait for welcome message
        response = await websocket.recv()
        data = json.loads(response)
        print(f"✅ {data.get('message', data)}\n")
        
        # Test messages
        test_messages = [
            "Hello Luna! How are you today?",
            "What can you tell me about ancient philosophy?",
            "That's fascinating! Tell me more about wisdom.",
            "Thank you for the conversation!"
        ]
        
        for msg in test_messages:
            print(f"👤 You: {msg}")
            
            # Send message
            await websocket.send(json.dumps({
                "message": msg,
                "session_id": "demo_session"
            }))
            
            # Receive response
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get('type') == 'message':
                print(f"🤖 Luna: {data['bot_response']}\n")
            elif data.get('type') == 'error':
                print(f"❌ Error: {data['message']}\n")
            
            await asyncio.sleep(1)  # Small delay between messages
        
        print("✨ Chat session complete!")

if __name__ == "__main__":
    try:
        asyncio.run(test_chat())
    except KeyboardInterrupt:
        print("\n👋 Chat ended by user")
    except Exception as e:
        print(f"❌ Error: {e}")
