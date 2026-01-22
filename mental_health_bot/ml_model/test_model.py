import joblib
import os
import sys

def test_model():
    model_path = os.path.join(os.path.dirname(__file__), 'emotion_model.pkl')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    try:
        model = joblib.load(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Test cases covering all categories
    test_cases = [
        ("I feel so alone and sad", "Sad/Depressed"),
        ("I am super excited about the news!", "Happy/Positive"),
        ("I am really worried about the exam", "Anxious/Stressed"),
        ("I am so angry at him", "Anger/Frustration"),
        ("I'm scared of the dark", "Fear/Panic"),
        ("I feel guilty for what I said", "Guilt/Shame"),
        ("I have no friends", "Lonely/Isolated"),
        ("I am just walking to the park", "Neutral"),
    ]

    print("\n--- Testing Model Predictions ---")
    correct = 0
    for text, expected in test_cases:
        prediction = model.predict([text])[0]
        match = "✅" if prediction == expected else "❌"
        if prediction == expected:
            correct += 1
        print(f"Text: '{text}'")
        print(f"  Predicted: {prediction}")
        print(f"  Expected:  {expected} {match}")
        print("-" * 30)

    accuracy = (correct / len(test_cases)) * 100
    print(f"\nTest Accuracy on specific cases: {accuracy:.1f}%")

    if accuracy < 100:
        print("\n⚠️  Note: Some predictions did not match expectations. This is normal for a small dataset.")
    else:
        print("\n🎉 All test cases passed!")

if __name__ == "__main__":
    test_model()
