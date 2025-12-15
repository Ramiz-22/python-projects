"""Hybrid LLM service: DeepSeek + Template Fallback"""
import os
import random
from typing import List, Dict
import openai

class LLMService:
    def __init__(self):
        self.initialized = False
        self.mode = "template"
        self.client = None      # For OpenAI/DeepSeek
        self.model_name = None  # For OpenAI/DeepSeek
        
    def initialize(self):
        """Initialize the service"""
        print("Initializing LLM service...")
        
        # Determine provider (Default to DeepSeek if configured, else Template)
        provider = os.getenv("AI_PROVIDER", "template").lower()
        
        # Auto-detect DeepSeek config
        if os.getenv("DEEPSEEK_API_KEY"):
            provider = "deepseek"
            
        print(f"Selected Provider: {provider}")
        
        if provider == "deepseek":
            self._init_deepseek()
        else:
            print("Using Template Mode (No AI Provider configured).")
            self.mode = "template"
            
        self.initialized = True
        print("Service initialized successfully!")

    def _init_deepseek(self):
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.ai/api/v1")
        model = os.getenv("DEEPSEEK_MODEL", "deepseek/deepseek-chat")
        
        if api_key:
            try:
                print(f"Connecting to DeepSeek via {base_url}...")
                self.client = openai.OpenAI(
                    base_url=base_url,
                    api_key=api_key
                )
                self.model_name = model
                self.mode = "deepseek"
                print(f"✅ Switched to REAL DeepSeek LLM Mode ({model})!")
            except Exception as e:
                print(f"❌ Failed to init DeepSeek: {e}")
                self.mode = "template"
        else:
            print("❌ Missing DEEPSEEK_API_KEY for deepseek provider.")
            self.mode = "template"

    def generate_response(
        self, 
        user_message: str, 
        character_personality: str,
        character_backstory: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> str:
        if not self.initialized:
            return "Service not initialized..."
        
        if self.mode == "deepseek":
            return self._generate_deepseek_response(user_message, character_personality, character_backstory, conversation_history)
        else:
            return self._generate_template_response(user_message, character_personality, character_backstory)

    def _generate_deepseek_response(self, message, personality, backstory, history):
        try:
            # Construct messages
            system_prompt = f"""You are a roleplay character.
Description: {backstory}
Personality: {personality}

Instructions:
- Reply naturally and strictly in character.
- Keep responses concise (under 3 sentences).
- Do NOT admit you are an AI.
"""
            messages = [{"role": "system", "content": system_prompt}]
            
            if history:
                for msg in history[-5:]:
                    messages.append({"role": "user", "content": msg.get('user', '')})
                    messages.append({"role": "assistant", "content": msg.get('bot', '')})
            
            messages.append({"role": "user", "content": message})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek Error: {e}")
            return f"I'm confused... (AI Error: {str(e)[:50]})"

    def _generate_template_response(self, user_message, character_personality, character_backstory):
        # ... (Keep existing simple dictionary logic for fallback) ...
        msg_lower = user_message.lower()
        if 'hello' in msg_lower: return "Hello!"
        if 'who are you' in msg_lower: return f"{character_backstory}"
        return "That's interesting."

# Singleton
llm_service = LLMService()
