"""Hybrid LLM service: DeepSeek + Gemini + Template Fallback"""
import os
import random
from typing import List, Dict
import google.generativeai as genai
import openai

class LLMService:
    def __init__(self):
        self.initialized = False
        self.mode = "template"
        self.model = None       # For Gemini
        self.client = None      # For OpenAI/DeepSeek
        self.model_name = None  # For OpenAI/DeepSeek
        
    def initialize(self):
        """Initialize the service"""
        print("Initializing LLM service...")
        
        # Determine provider
        provider = os.getenv("AI_PROVIDER", "gemini").lower()
        print(f"Selected Provider: {provider}")
        
        if provider == "deepseek":
            self._init_deepseek()
        elif provider == "gemini":
            self._init_gemini()
        else:
            print(f"Unknown provider {provider}, using templates.")
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

    def _init_gemini(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Dynamic model selection
                print("Listing available Gemini models...")
                available = []
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            available.append(m.name)
                except:
                    print("Could not list models (network/auth issue).")
                
                if not available:
                    # Fallback default if list failed but key exists
                    model_name = 'gemini-1.5-flash'
                else:
                    # Auto select
                    model_name = next((m for m in available if 'gemini-1.5-flash' in m), None)
                    if not model_name: model_name = next((m for m in available if 'gemini-pro' in m), None)
                    if not model_name: model_name = available[0]
                
                print(f"✅ Auto-selected Gemini Model: {model_name}")
                self.model = genai.GenerativeModel(model_name)
                self.mode = "gemini"
                print("✅ Switched to REAL Gemini LLM Mode!")
            except Exception as e:
                print(f"❌ Failed to init Gemini: {e}")
                self.mode = "template"
        else:
            print("⚠️ No GOOGLE_API_KEY found.")
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
        elif self.mode == "gemini":
            return self._generate_gemini_response(user_message, character_personality, character_backstory, conversation_history)
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
            return f"I'm confused... (DeepSeek Error: {str(e)[:50]})"

    def _generate_gemini_response(self, message, personality, backstory, history):
        # ... (Same Gemini logic as before) ...
        try:
            conversation_text = ""
            if history:
                for msg in history[-5:]:
                    conversation_text += f"User: {msg.get('user', '')}\nYou: {msg.get('bot', '')}\n"
            
            prompt = f"""You are a roleplay character.
Description: {backstory}
Personality: {personality}
Instructions: Stay in character. Concise responses.
Previous Conversation:
{conversation_text}

User: {message}
You:"""
            
            response = self.model.generate_content(prompt)
            return response.text.strip() if response.text else "..."
        except Exception as e:
            print(f"Gemini error: {e}")
            return f"Gemini Error: {str(e)[:50]}"

    def _generate_template_response(self, user_message, character_personality, character_backstory):
         # ... (Keep existing simple dictionary logic) ...
        msg_lower = user_message.lower()
        if 'hello' in msg_lower: return f"Hello! {self._personality_context(character_personality)}"
        if 'who are you' in msg_lower: return f"{character_backstory}"
        return f"That's interesting. {self._personality_context(character_personality)}"

    # Helper methods (Minimally defined for template fallback)
    def _personality_context(self, p): return "I'm here to chat!"
    def _backstory_context(self, b): return "Nice to meet you."
    def _personality_response(self, p, c): return "I see."
    def _contextual_response(self, m, p): return "Tell me more."
    def _empathy_response(self, p): return "I understand."

# Singleton
llm_service = LLMService()
