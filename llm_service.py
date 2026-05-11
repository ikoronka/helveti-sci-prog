import google.generativeai as genai
import os


class LLMService:
    def __init__(self):
        # API key should be set in the environment
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def generate_conclusion(self, corr, p_value):
        prompt = f"Analyze this scientific result: Correlation between Rent and Size is {corr} with a p-value of {p_value}. Provide a short research conclusion."
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Could not generate conclusion: {e}"
