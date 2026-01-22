import json
import numpy as np
import re
import os
import math

class SGDInference:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'model_params.json')
        
        with open(model_path, 'r') as f:
            self.model_data = json.load(f)
            
        self.vocab = self.model_data['vocabulary']
        self.idf = np.array(self.model_data['idf'])
        self.coef = np.array(self.model_data['coef'])
        self.intercept = np.array(self.model_data['intercept'])
        self.classes = self.model_data['classes']
        
        # Regex for tokenization (same as CountVectorizer default)
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")

    def preprocess(self, text):
        return self.token_pattern.findall(text.lower())

    def get_ngrams(self, tokens, n=2):
        # Generate unigrams
        ngrams = tokens[:]
        # Generate bigrams
        if n >= 2:
            for i in range(len(tokens) - 1):
                ngrams.append(f"{tokens[i]} {tokens[i+1]}")
        return ngrams

    def predict(self, text):
        tokens = self.preprocess(text)
        ngrams = self.get_ngrams(tokens)
        
        # Count vectorizer
        # We only care about terms in the vocab
        term_counts = {}
        for term in ngrams:
            if term in self.vocab:
                idx = self.vocab[term]
                term_counts[idx] = term_counts.get(idx, 0) + 1
                
        if not term_counts:
            # Fallback if no known words found
            return self.classes[0] # Default to first class or handle gracefully

        # Create feature vector (sparse representation)
        indices = list(term_counts.keys())
        counts = list(term_counts.values())
        
        # Apply TF-IDF
        # TF * (IDF + 1)  (assuming smooth_idf=True which is default)
        # Note: Scikit-learn TfidfTransformer with norm='l2'
        
        # We construct the sparse vector values
        values = []
        for idx, count in zip(indices, counts):
            # tf = count (raw count)
            # idf = self.idf[idx]
            val = count * (self.idf[idx]) # Wait, sklearn stores idf_ as log(N/df) + 1 if smooth_idf.
            # Actually, let's check what idf_ contains.
            # In TfidfTransformer, idf_ is precomputed.
            # transform(X) -> X * idf_
            # So it is just count * idf_ value from the array.
            values.append(val)
            
        values = np.array(values)
        
        # L2 Normalization
        norm = np.linalg.norm(values)
        if norm > 0:
            values = values / norm
            
        # Prediction: X @ coef.T + intercept
        # Since X is sparse, we only sum the relevant columns of coef
        
        # scores = np.dot(X, coef.T) + intercept
        # X has non-zero values at 'indices' with 'values'
        
        # Efficient sparse dot product
        # scores[class_k] = sum(X[i] * coef[class_k, i]) + intercept[class_k]
        
        scores = np.zeros(len(self.classes))
        for k in range(len(self.classes)):
            dot_product = 0
            for i, val in zip(indices, values):
                dot_product += val * self.coef[k][i]
            scores[k] = dot_product + self.intercept[k]
            
        best_class_idx = np.argmax(scores)
        return self.classes[best_class_idx]

if __name__ == "__main__":
    # Test
    predictor = SGDInference()
    tests = [
        "I feel really down and hopeless",
        "I am so happy and excited today!",
        "I am really worried about the future",
        "I am furious with him"
    ]
    for t in tests:
        print(f"'{t}' => {predictor.predict(t)}")
