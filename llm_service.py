import ollama


class LLMService:
    def generate_conclusion(self, corr, p_value):
        prompt = f"Analyze this scientific result: Correlation between Rent and Size is {corr} with a p-value of {p_value}. Provide a short research conclusion."
        try:
            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            return response["message"]["content"]
        except Exception as e:
            return f"Could not generate conclusion: {e}"
