# PsycheFlow: AI Mental Health Companion - Presentation Content

## 1. Problem Statement
*   **Lack of Accessibility**: Professional mental health support is often expensive and not immediately available during moments of crisis.
*   **Social Stigma**: Many individuals hesitate to seek help due to fear of judgment or societal stigma surrounding mental health.
*   **Time Constraints**: Therapy sessions are scheduled, but emotional distress can happen at any time, day or night.
*   **Limited Self-Help Tools**: Existing chatbots often lack empathy or context, providing generic responses that fail to address the user's specific emotional state.

## 2. Proposed System/Solution
*   **PsycheFlow**: An intelligent, accessible, and judgment-free mental health companion chatbot.
*   **Real-Time Support**: Available 24/7 to listen, analyze emotions, and provide supportive, empathetic feedback.
*   **Emotion-Aware**: Unlike standard chatbots, PsycheFlow uses a custom Machine Learning model to detect the user's underlying emotion (e.g., Joy, Sadness, Anger, Fear).
*   **Hybrid AI Approach**: Combines the speed and privacy of local ML classification with the generative power of Large Language Models (LLM) for natural conversation.
*   **Privacy-First**: Designed to run locally, ensuring user data remains private and secure.

## 3. System Development Approach (Technology Used)
*   **Backend Framework**: **Django** (Python) - Robust, secure, and scalable web framework for managing chat logic and API integration.
*   **Frontend**: **HTML5, CSS3, JavaScript** - Clean, calming user interface with real-time dynamic updates (AJAX).
*   **Machine Learning**:
    *   **Scikit-learn**: For model training and pipeline creation.
    *   **Pandas**: For dataset manipulation and preprocessing.
    *   **Joblib**: For model serialization and loading.
*   **Generative AI**: **Google Gemini API (gemini-2.0-flash)** - Used to generate human-like, contextually appropriate responses based on the detected emotion.
*   **Dataset**: Kaggle "Emotions" dataset (nelgiriyewithana/emotions) containing ~416k labeled text samples.

## 4. Algorithm & Deployment
### Algorithm: Emotion Detection Pipeline
1.  **Input Processing**: User text is cleaned (lowercased, special characters removed).
2.  **Vectorization**: **CountVectorizer** (N-grams 1,2) converts text into numerical token counts.
3.  **Transformation**: **TfidfTransformer** (Term Frequency-Inverse Document Frequency) weights the tokens to highlight important words.
4.  **Classification**: **SGDClassifier** (Stochastic Gradient Descent) with `loss='log_loss'` is used for efficient, linear classification into 6 emotion categories (Sadness, Joy, Love, Anger, Fear, Surprise).
5.  **Context Management**: A session-based memory system tracks the previous emotion to prevent abrupt context switching (e.g., maintaining a supportive tone if the user was just angry).
6.  **Safety Nets**: Hardcoded overrides for high-risk keywords (e.g., self-harm, extreme aggression) to ensure safe responses.

### Deployment
*   **Local Host**: Currently deployed on a local development server (`127.0.0.1:8000`).
*   **Environment Management**: Uses `python-dotenv` for secure API key management.
*   **Scalability**: Ready for deployment on cloud platforms like AWS, Heroku, or Azure using WSGI/ASGI servers.

## 5. Result
*   **Accurate Classification**: The system successfully identifies user emotions with high confidence.
*   **Empathetic Interaction**: The integration of Gemini API allows for nuanced, "human-like" conversations rather than robotic scripts.
*   **Visual Feedback**: The UI displays a "Mood Badge" that updates in real-time, helping users become more self-aware of their emotional state.
*   **Robustness**: Successfully handles edge cases (e.g., short questions, contradictory statements) through implemented safety nets.

## 6. Conclusion
*   PsycheFlow successfully demonstrates how AI can bridge the gap between clinical therapy and daily self-care.
*   By combining traditional Machine Learning for classification with Generative AI for interaction, the system offers a balanced solution that is both accurate and empathetic.
*   It provides a critical "first line of defense" for mental well-being, offering immediate support to those who might otherwise go unheard.

## 7. Future Scope
*   **Voice Integration**: Adding Speech-to-Text (STT) and Text-to-Speech (TTS) for a hands-free conversational experience.
*   **User Accounts & History**: Implementing secure login to track emotional trends over weeks or months.
*   **Professional Escalation**: Detecting crisis situations and automatically providing helpline numbers or connecting to human therapists.
*   **Mobile Application**: Converting the web app into a React Native or Flutter mobile app for on-the-go support.

## 8. References
1.  **Django Documentation**: https://docs.djangoproject.com/
2.  **Scikit-learn Documentation**: https://scikit-learn.org/stable/
3.  **Google AI Studio (Gemini)**: https://ai.google.dev/
4.  **Kaggle Dataset**: https://www.kaggle.com/datasets/nelgiriyewithana/emotions
