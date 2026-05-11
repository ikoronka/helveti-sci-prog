# Scientific House Rent Dashboard

This project is an MVP Scientific Programming dashboard designed to analyze the [House Rent Prediction Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset/data).

## Features
- **Automated Data Pipeline:** Downloads the dataset directly from Kaggle and processes it into a local SQLite database.
- **Interactive Dashboard:** Built with [Marimo](https://marimo.io/), featuring reactive filters for City and BHK (Bedroom, Hall, Kitchen).
- **Statistical Analysis:** Performs regression and correlation analysis between Rent and Size using `scipy`.
- **AI-Driven Insights:** Generates research conclusions based on your filtered analysis using Google Gemini.

## Prerequisites
- [uv](https://github.com/astral-sh/uv) (for package management)
- A [Kaggle API Key](https://www.kaggle.com/settings)
- A [Google Gemini API Key](https://aistudio.google.com/)

## Setup
1. **API Keys:**
   - Place your `kaggle.json` in `~/.kaggle/kaggle.json`.
   - Add your Google API key to your environment variables:
     ```bash
     export GOOGLE_API_KEY="your_api_key_here"
     ```
2. **Dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```
3. **Run the Dashboard:**
   ```bash
   marimo edit app.py
   ```
