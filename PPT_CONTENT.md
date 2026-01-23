# Luma: AI Mental Health Companion - Presentation Content

## 1. Problem Statement
*   **Lack of Accessibility**: Professional mental health support is often expensive and not immediately available during moments of crisis.
*   **Social Stigma**: Many individuals hesitate to seek help due to fear of judgment or societal stigma surrounding mental health.
*   **Time Constraints**: Therapy sessions are scheduled, but emotional distress can happen at any time, day or night.
*   **Limited Self-Help Tools**: Existing chatbots often lack empathy or context, providing generic responses that fail to address the user's specific emotional state.

## 2. Proposed System/Solution
*   **Luma**: An intelligent, accessible, and judgment-free mental health companion chatbot.
*   **Real-Time Support**: Available 24/7 to listen, analyze emotions, and provide supportive, empathetic feedback.
*   **Emotion-Aware**: Unlike standard chatbots, Luma uses a custom optimized Machine Learning model to detect the user's underlying emotion (e.g., Joy, Sadness, Anger, Fear).
*   **Hybrid AI Approach**: Combines the speed of a custom, lightweight ML classifier with the generative power of Large Language Models (LLM) for natural conversation.
*   **Lightweight & Scalable**: Migrated from a heavy web framework to a streamlined Streamlit application for faster performance and easier deployment.

## 3. System Development Approach (Technology Used)
*   **Framework**: **Streamlit** (Python) - Fast, data-centric web framework for building interactive AI applications.
*   **Frontend**: **Streamlit UI** - Clean, responsive chat interface with real-time state management.
*   **Machine Learning (Custom Implementation)**:
    *   **NumPy**: For high-performance numerical operations and matrix calculations.
    *   **Custom SGD Inference**: A pure-Python/NumPy implementation of the classifier to remove heavy dependencies (scikit-learn) for deployment.
*   **Generative AI**: **Google Gemini API (gemini-1.5-flash)** - Used to generate human-like, contextually appropriate responses based on the detected emotion.
*   **Dataset**: Kaggle "Emotions" dataset (nelgiriyewithana/emotions) containing ~416k labeled text samples.

## 4. Algorithm & Deployment
### Algorithm: Emotion Detection Pipeline
1.  **Input Processing**: User text is cleaned (lowercased, special characters removed) using regular expressions.
2.  **Vectorization**: **Custom N-gram Generation** converts text into token counts without external libraries.
3.  **Transformation**: **TF-IDF Weighting** (Term Frequency-Inverse Document Frequency) is applied using pre-calculated model parameters.
4.  **Classification**: **SGDClassifier (Linear SVM/LogReg)** logic implemented in pure NumPy for efficient classification into 6 emotion categories (Sadness, Joy, Love, Anger, Fear, Surprise).
5.  **Context Management**: A session-based memory system tracks the previous emotion to prevent abrupt context switching.
6.  **Safety Nets**: Intelligent overrides for high-risk keywords and context preservation (e.g., preventing "Joy" triggers from short, ambiguous inputs unless explicitly positive).

### Deployment
*   **Platform**: **Streamlit Cloud** (or local hosting).
*   **Optimization**: Model size reduced from ~87MB to <1MB by feature selection and JSON serialization, solving serverless deployment constraints.
*   **Environment Management**: Uses `python-dotenv` for secure API key management.

## 5. Result
*   **Accurate Classification**: The system successfully identifies user emotions with high confidence using a lightweight custom model.
*   **Empathetic Interaction**: The integration of Gemini API allows for nuanced, "human-like" conversations rather than robotic scripts.
*   **Real-Time Feedback**: The UI displays a "Mood Detected" indicator that updates instantly.
*   **Robustness**: Successfully handles edge cases (e.g., short questions, contradictory statements) through implemented safety nets.

## 6. Conclusion
*   Luma successfully demonstrates how AI can bridge the gap between clinical therapy and daily self-care.
*   By moving from a heavy framework to a lightweight, optimized architecture, the project proves that powerful AI tools can be made accessible and efficient.
*   It provides a critical "first line of defense" for mental well-being, offering immediate support to those who might otherwise go unheard.

## 7. Future Scope
*   **Voice Integration**: Adding Speech-to-Text (STT) and Text-to-Speech (TTS) for a hands-free conversational experience.
*   **User Accounts & History**: Implementing secure login to track emotional trends over weeks or months.
*   **Professional Escalation**: Detecting crisis situations and automatically providing helpline numbers or connecting to human therapists.
*   **Mobile Application**: Converting the web app into a React Native or Flutter mobile app for on-the-go support.

## 8. References
*   **Streamlit Documentation**: https://docs.streamlit.io/
*   **NumPy Documentation**: https://numpy.org/doc/
*   **Google AI Studio (Gemini)**: https://ai.google.dev/
*   **Kaggle Dataset**: https://www.kaggle.com/datasets/nelgiriyewithana/emotions
