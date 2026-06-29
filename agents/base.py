import google.generativeai as genai
from config.settings import Settings
from utils.logging import logger

class BaseAgent:
    def __init__(self, name: str, system_instruction: str):
        """
        Initializes the agent with a name and a system instruction.
        """
        self.name = name
        self.system_instruction = system_instruction
        self.model = None
        
    def init_model(self):
        """
        Initializes the Gemini model using the API key from Settings.
        Should be called before invoking the agent.
        """
        if not Settings.GEMINI_API_KEY:
            logger.error(f"[{self.name}] API key is missing. Model cannot be initialized.")
            raise ValueError("Google Gemini API Key is missing. Please set it in your configuration.")
        
        try:
            logger.info(f"[{self.name}] Initializing Gemini model '{Settings.GEMINI_MODEL}'...")
            genai.configure(api_key=Settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=Settings.GEMINI_MODEL,
                system_instruction=self.system_instruction
            )
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize Gemini model: {e}")
            raise e

    def call_llm(self, prompt: str, json_mode: bool = False) -> str:
        """
        Sends a prompt to the model and returns the response.
        Handles API errors gracefully and logs attempts.
        """
        if not self.model:
            self.init_model()

        logger.info(f"[{self.name}] Sending prompt to model...")
        try:
            generation_config = {}
            if json_mode:
                generation_config = {"response_mime_type": "application/json"}
                
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            if not response.text:
                raise ValueError("Received empty response from LLM.")
                
            logger.info(f"[{self.name}] LLM call successful.")
            return response.text
        except Exception as e:
            logger.error(f"[{self.name}] Error during LLM call: {e}")
            raise e
