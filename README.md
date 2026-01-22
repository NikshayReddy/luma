# Luma

Luma is an intelligent mental health companion designed to help you navigate the complex currents of your emotions. By combining advanced sentiment analysis with compassionate AI, Luma provides a safe, judgment-free space to express your thoughts, track your emotional well-being, and receive supportive feedback in real-time. Whether you're feeling overwhelmed, joyful, or somewhere in between, Luma is here to listen and help you find your balance.

## 🚀 Features

*   **Emotion Detection**: Uses a TF-IDF + SGDClassifier model to classify text into 6 emotions: Sadness, Joy, Love, Anger, Fear, Surprise.
*   **Empathetic AI Responses**: Integrated with Google Gemini API for natural, supportive conversations.
*   **Context Awareness**: Remembers conversation context (e.g., if you were angry, it won't suddenly become happy if you ask a short question).
*   **Safety Nets**: Hardcoded safety overrides for negative keywords to prevent inappropriate positive responses to distress.
*   **Privacy-First**: Runs locally; the ML model is embedded in the application.

## 🛠️ Installation

### 1. Prerequisites
*   Python 3.10 or higher
*   Git (optional, for cloning)

### 2. Setup the Environment

Open your terminal (PowerShell or CMD) and navigate to the project folder.

1.  **Create a virtual environment** (Recommended):
    ```bash
    python -m venv .venv
    ```

2.  **Activate the virtual environment**:
    *   **Windows**:
        ```bash
        .\.venv\Scripts\Activate
        ```
    *   **Mac/Linux**:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Configure API Keys

1.  Create a file named `.env` in the root directory (same place as `manage.py` and `requirements.txt`).
2.  Add your Google Gemini API key to it:

    ```env
    GOOGLE_API_KEY=your_actual_api_key_here
    ```
    *(Note: The chatbot will still work without this key using the local ML model and pre-written responses, but the quality will be lower.)*

## ▶️ How to Run

### Option 1: Quick Start (Windows)
Double-click the `run_bot.bat` file in the project folder. This will automatically start the server.

### Option 2: Manual Start (Terminal)
Run the following command in your terminal (ensure your virtual environment is activated):

```bash
cd mental_health_bot
python manage.py runserver
```

Once the server is running, open your web browser and go to:
👉 **http://127.0.0.1:8000/**

## 🧠 Model & Data
*   **Dataset**: [Emotions Dataset by Nelgiriyewithana](https://www.kaggle.com/datasets/nelgiriyewithana/emotions) (15.7M text samples).
*   **Algorithm**: Stochastic Gradient Descent (SGD) Classifier with TF-IDF Vectorization.
*   **Accuracy**: High efficiency for real-time text classification.

## 📂 Project Structure
*   `mental_health_bot/`: Main Django project folder.
    *   `chatbot/`: App containing views, URLs, and HTML templates.
    *   `ml_model/`: Contains the trained `emotion_model.pkl` and training scripts.
*   `requirements.txt`: Python dependencies.
*   `.env`: Configuration file for API keys (hidden).

## ⚠️ Disclaimer
This chatbot is an AI tool for emotional support and is **not** a substitute for professional mental health advice. If you or someone you know is in crisis, please seek professional help immediately.
