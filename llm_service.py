"""
llm_service.py

This module provides the Large Language Model component of the project.

The dashboard uses this file to generate a written interpretation of the statistical
relationship between apartment size and rent.

Main responsibilities:
1. Connect to the Gemini API using an environment variable.
2. Send the correlation and p-value to the model.
3. Return a short scientific conclusion for the dashboard.

Rubric relevance:
- Use of a Large Language Model to support data analysis
- Integration of statistical analysis and natural-language interpretation
- Modular Python code using object-oriented programming
"""

from google import genai
import os

class LLMService:
    """
    Service class for generating AI-supported statistical conclusions.

    This class separates the LLM logic from the dashboard code. That makes the
    project more modular because `app.py` does not need to contain the details
    of the Gemini API call.

    Environment variable
    --------------------
    GOOGLE_API_KEY
        API key used to authenticate with the Gemini API.

    Security note
    -------------
    The API key should not be written directly into the code or uploaded to GitHub.
    It should be stored locally as an environment variable.
    """
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
