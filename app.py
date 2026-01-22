import streamlit as st
import os
import random
import google.generativeai as genai
from dotenv import load_dotenv
from mental_health_bot.ml_model.inference import SGDInference

# Page Config
st.set_page_config(
    page_title="Luma - Mental Health Companion",
    page_icon="✨",
    layout="centered"
)

# Load Environment Variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in environment variables. Please set it to use the chatbot.")
    st.stop()

genai.configure(api_key=api_key)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! I am here to listen, support, and chat with you. How are you feeling today?"
    }]
if "last_emotion" not in st.session_state:
    st.session_state.last_emotion = None

# Load ML Model (Cached Resource)
@st.cache_resource
def load_model():
    try:
        # Construct path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, 'mental_health_bot', 'ml_model', 'model_params.json')
        return SGDInference(model_path)
    except Exception as e:
        st.error(f"Error loading ML model: {e}")
        return None

emotion_model = load_model()

# Constants
EMOTION_MAP = {
    0: 'Sadness', 1: 'Joy', 2: 'Love', 
    3: 'Anger', 4: 'Fear', 5: 'Surprise'
}

RESPONSES = {
    'Sadness': [
        "I'm sorry to hear that you're feeling down. I'm here to listen.",
        "It's okay to feel sad sometimes. Do you want to talk about it?",
        "Sending you a virtual hug. You are not alone.",
        "Take your time. I'm here for you.",
        "Is there anything specific that's making you feel this way?"
    ],
    'Joy': [
        "I'm listening. Tell me more.",
        "That sounds positive.",
        "Keep going, I'm listening.",
        "I see. How does that make you feel?",
        "Glad to hear it."
    ],
    'Love': [
        "That sounds lovely.",
        "It's good to have things we care about.",
        "Tell me more about that.",
        "I'm listening.",
        "Love is important."
    ],
    'Anger': [
        "I hear your frustration. It's okay to let it out.",
        "Take a deep breath. I'm here to listen to your side.",
        "It sounds like you're going through a tough time.",
        "Anger is a valid emotion. Do you want to vent?",
        "I'm listening. Tell me what happened."
    ],
    'Fear': [
        "It's okay to be scared. You are safe here.",
        "Take a deep breath. We can get through this together.",
        "I'm here with you. You don't have to face this alone.",
        "What can I do to help you feel more comfortable?",
        "Focus on the present moment. You are okay."
    ],
    'Surprise': [
        "Wow! That sounds unexpected!",
        "Life is full of surprises, isn't it?",
        "That's quite a turn of events!",
        "Tell me more about it!",
        "How do you feel about this surprise?"
    ]
}

DEFAULT_RESPONSES = [
    "I'm listening.", "Tell me more.", 
    "I'm here for you.", "Go on, I'm listening."
]

def get_bot_response(user_input):
    detected_emotion_label = "Unknown"
    
    # 1. ML Prediction
    if emotion_model:
        try:
            detected_emotion_id = emotion_model.predict(user_input)
            detected_emotion_label = EMOTION_MAP.get(detected_emotion_id, "Unknown")
        except Exception:
            pass

    # 2. Safety Logic
    # Negative Keyword Safety Net
    negative_keywords = ['demote', 'fire', 'hate', 'stupid', 'idiot', 'kill', 'die', 'angry', 'furious', 'mad', 'boss', 'bad', 'terrible', 'hit', 'punch', 'hurt', 'hell', 'damn', 'wtf']
    if any(keyword in user_input.lower() for keyword in negative_keywords):
        if detected_emotion_label in ['Joy', 'Love', 'Surprise']:
            detected_emotion_label = 'Anger'

    # Question/Neutral Safety Net
    question_keywords = ['what', 'how', 'why', 'when', 'where', 'who', '?']
    is_question = any(k in user_input.lower() for k in question_keywords)
    # Only treat as "too short to be meaningful" if it's REALLY short (e.g., 1-2 words) AND not a clear strong emotion
    is_short = len(user_input.split()) < 3
    
    # Check for strong positive keywords that should override "shortness"
    positive_keywords = ['happy', 'good', 'great', 'love', 'excellent', 'amazing', 'wonderful', 'joy', 'excited', 'better', 'fine', 'ok', 'okay']
    has_positive = any(k in user_input.lower() for k in positive_keywords)

    if (is_question or (is_short and not has_positive)) and detected_emotion_label in ['Joy', 'Surprise']:
        if st.session_state.last_emotion in ['Anger', 'Sadness', 'Fear']:
            # If asking a question or giving a very short non-positive answer, assume context continues
            detected_emotion_label = st.session_state.last_emotion
        else:
            # Otherwise, treat as Neutral
            detected_emotion_label = 'Neutral'

    # Update Session State
    if detected_emotion_label != 'Neutral':
        st.session_state.last_emotion = detected_emotion_label

    # 3. Generate Response (Gemini -> Local Fallback)
    bot_response = ""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = (
            f"You are a compassionate mental health companion named Luma. "
            f"The user says: '{user_input}'. "
            f"My internal emotion detection model has identified the user's emotion as '{detected_emotion_label}'. "
            f"Please analyze the text yourself. If the text clearly conveys a different emotion "
            f"that contradicts the internal label, prioritize your own analysis. "
            f"Provide a supportive, empathetic response (max 2-3 sentences) appropriate for the user's actual emotion. "
            f"Do not explicitly mention the internal label."
        )
        response = model.generate_content(prompt)
        if response and response.text:
            bot_response = response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")

    if not bot_response:
        if detected_emotion_label in RESPONSES:
            bot_response = random.choice(RESPONSES[detected_emotion_label])
        else:
            bot_response = random.choice(DEFAULT_RESPONSES)
            
    return bot_response, detected_emotion_label

# UI Layout
st.title("✨ Luma")
st.markdown("Your guiding light through emotional currents")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How are you feeling?"):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_text, emotion = get_bot_response(prompt)
            st.markdown(response_text)
            if emotion and emotion != "Unknown":
                st.caption(f"Mood detected: {emotion}")
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
