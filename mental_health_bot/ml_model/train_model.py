import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# Define paths
current_dir = os.path.dirname(__file__)
data_path = os.path.join(current_dir, 'data', 'text.csv')
model_path = os.path.join(current_dir, 'emotion_model.pkl')

def train_model():
    print("Loading dataset...")
    try:
        # Check if data file exists
        if not os.path.exists(data_path):
            print(f"Error: Data file not found at {data_path}")
            return

        df = pd.read_csv(data_path)
        # Verify columns
        if 'text' not in df.columns or 'label' not in df.columns:
            print("Columns found:", df.columns)
            if 'text' not in df.columns:
                text_col = [c for c in df.columns if 'text' in c.lower()]
                if text_col:
                    df.rename(columns={text_col[0]: 'text'}, inplace=True)
            if 'label' not in df.columns:
                label_col = [c for c in df.columns if 'label' in c.lower()]
                if label_col:
                    df.rename(columns={label_col[0]: 'label'}, inplace=True)
        
        X = df['text']
        y = df['label']
        
        print(f"Training on {len(df)} samples...")
        
        # Create Pipeline with reduced max_features for smaller model size
        text_clf = Pipeline([
            ('vect', CountVectorizer(ngram_range=(1, 2), max_features=5000)),
            ('tfidf', TfidfTransformer()),
            ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
        ])
        
        text_clf.fit(X, y)
        
        # Save Model
        joblib.dump(text_clf, model_path)
        print(f"Model saved to {model_path}")
        
        # Test a few examples
        test_sentences = [
            "I feel really down and hopeless",
            "I am so happy and excited today!",
            "I am really worried about the future",
            "I am furious with him",
            "I feel so romantic and loved",
            "Wow, I didn't expect that!"
        ]
        
        predicted = text_clf.predict(test_sentences)
        
        emotion_map = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        
        for text, label in zip(test_sentences, predicted):
            print(f"'{text}' => {emotion_map.get(label, label)}")
            
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    train_model()
