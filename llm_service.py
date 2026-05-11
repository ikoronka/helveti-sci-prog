from google import genai
import os


class LLMService:
    def __init__(self):
        # The SDK automatically looks for GEMINI_API_KEY environment variable
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def generate_conclusion(self, corr, p_value):
        prompt = f"Analyze this scientific result: Correlation between Rent and Size is {corr} with a p-value of {p_value}. Provide a short research conclusion."
        try:
            # Using the gemini-2.0-flash-lite model
            response = self.client.models.generate_content(
                model="gemini-flash-lite-latest", contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Could not generate conclusion: {e}"
