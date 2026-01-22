from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import ensure_csrf_cookie
import os
import random
import google.generativeai as genai
from dotenv import load_dotenv
from ml_model.inference import SGDInference

# Load Environment Variables
# Assuming .env is in the project root (one level up from manage.py, or two levels up from here)
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load ML Model
try:
    emotion_model = SGDInference()
    print("ML Model loaded successfully")
except Exception as e:
    print(f"Error loading ML model: {e}")
    emotion_model = None

# Emotion Labels Mapping (nelgiriyewithana/emotions)
EMOTION_MAP = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# Predefined Responses based on Emotion
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
    "I'm listening.",
    "Tell me more.",
    "I'm here for you.",
    "Go on, I'm listening."
]

@ensure_csrf_cookie
def chat_view(request):
    return render(request, 'chatbot/index.html')

def get_response(request):
    if request.method == 'POST':
        user_input = request.POST.get('message')
        
        try:
            # Predict Emotion using ML Model
            detected_emotion_label = "Unknown"
            detected_emotion_id = -1
            
            if emotion_model:
                detected_emotion_id = emotion_model.predict(user_input)
                # detected_emotion_id is already an integer (0-5)
                detected_emotion_label = EMOTION_MAP.get(detected_emotion_id, "Unknown")

            # --- CONTEXT & SAFETY LOGIC START ---
            
            # 1. Retrieve previous emotion from session
            last_emotion = request.session.get('last_emotion', None)

            # 2. Negative Keyword Safety Net (Overrides ML)
            negative_keywords = ['demote', 'fire', 'hate', 'stupid', 'idiot', 'kill', 'die', 'angry', 'furious', 'mad', 'boss', 'bad', 'terrible', 'hit', 'punch', 'hurt', 'hell', 'damn', 'wtf']
            if any(keyword in user_input.lower() for keyword in negative_keywords):
                if detected_emotion_label in ['Joy', 'Love', 'Surprise']:
                    detected_emotion_label = 'Anger' # Force Anger for negative keywords

            # 3. Question/Neutral Safety Net (Overrides Joy/Surprise for short questions)
            question_keywords = ['what', 'how', 'why', 'when', 'where', 'who', '?']
            is_question = any(k in user_input.lower() for k in question_keywords)
            is_short = len(user_input.split()) < 8
            
            if (is_question or is_short) and detected_emotion_label in ['Joy', 'Surprise']:
                # If we were previously talking about something negative, assume we still are
                if last_emotion in ['Anger', 'Sadness', 'Fear']:
                    detected_emotion_label = last_emotion
                else:
                    # Otherwise, treat as Neutral (force fallback)
                    detected_emotion_label = 'Neutral'

            # 4. Save current emotion to session for next turn
            if detected_emotion_label != 'Neutral':
                 request.session['last_emotion'] = detected_emotion_label
            
            # --- CONTEXT & SAFETY LOGIC END ---

            # Try Gemini API First
            bot_response = ""
            try:
                # Using gemini-1.5-flash for better stability/quota
                model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = (
                    f"You are a compassionate mental health companion. "
                    f"The user says: '{user_input}'. "
                    f"My internal emotion detection model has identified the user's emotion as '{detected_emotion_label}'. "
                    f"However, please analyze the text yourself. If the text clearly conveys a different emotion "
                    f"(especially negative ones like anger or sadness) that contradicts the internal label, "
                    f"prioritize your own analysis and provide a supportive, empathetic response (max 2-3 sentences) "
                    f"appropriate for the actual emotion."
                )
                response = model.generate_content(prompt)
                if response and response.text:
                    bot_response = response.text
                    print(f"Gemini Response Generated: {bot_response}")
            except Exception as e:
                print(f"Gemini API Error: {e}")
                bot_response = ""

            # Fallback to Local Responses if Gemini fails or returns empty
            if not bot_response:
                if detected_emotion_label in RESPONSES:
                    bot_response = random.choice(RESPONSES[detected_emotion_label])
                else:
                    bot_response = random.choice(DEFAULT_RESPONSES)

            return JsonResponse({'response': bot_response, 'emotion': detected_emotion_label})
            
        except Exception as e:
            print(f"Error processing request: {e}")
            return JsonResponse({'response': "I'm having trouble processing that right now. Can we try again?", 'emotion': 'Neutral'})

    return JsonResponse({'response': 'Invalid request', 'emotion': 'Neutral'})
