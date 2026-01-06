import os
import re
import time
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLMClient:
    """
    Model client designed for LLM evaluation.
    Supports Hugging Face Router API with multi-dimensional result extraction.
    """
    def __init__(self):
        self.api_token = os.getenv("HF_TOKEN")
        self.base_url = os.getenv("HF_API_BASE", "https://router.huggingface.co/v1")
        self.model_name = os.getenv("DEFAULT_MODEL", "deepseek-ai/DeepSeek-V3.2:novita")
        
        if not self.api_token:
            raise ValueError("Error: Please configure HF_TOKEN in .env file")
            
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_token
        )

    def get_response(self, prompt, max_retries=3):
        """
        Core inference function with retry mechanism and parameter control.
        """
        for i in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a precise assistant for academic evaluation. Provide the final answer clearly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,  # Set to 0 for reproducibility in evaluation
                    max_tokens=500
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                time.sleep(2)  # Wait on rate limiting or network issues
        return None

    def extract_final_answer(self, text):
        """
        Evaluation core: Extract final answer from model's verbose response.
        E.g., extract "42" from "After calculation, the answer should be 42."
        """
        if not text:
            return ""
        
        # Match last number or specific format
        patterns = [
            r"The answer is ([\d\.]+)",          # Match specific phrase
            r"#### ([\d\.]+)",                   # Match GSM8K common format
            r"(\d+)(?=\D*$)"                     # Match last number in text
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
        return text.strip()
