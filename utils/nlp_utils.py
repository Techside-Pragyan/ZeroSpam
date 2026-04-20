import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import numpy as np

# Ensure NLTK data is available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

def clean_text(text):
    # Convert to lowercase
    text = str(text).lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = nltk.word_tokenize(text)
    # Remove stopwords and stemming
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    cleaned_tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return cleaned_tokens

def explain_prediction(text, model, vectorizer):
    # Clean and tokenize
    tokens = clean_text(text)
    if not tokens:
        return []
        
    # Transform to TF-IDF
    tfidf_vec = vectorizer.transform([" ".join(tokens)])
    
    # Get feature names and weights
    feature_names = vectorizer.get_feature_names_out()
    feature_weights = model.coef_[0]
    
    # Map tokens back to their weights
    explanations = []
    text_processed = " ".join(tokens)
    
    # We want to see which words in the ORIGINAL text contributed most
    # This is a bit simplified: we'll look for words that, when stemmed, 
    # match the features with high weights.
    
    for word in set(re.findall(r'\b\w+\b', text.lower())):
        stemmed = PorterStemmer().stem(word)
        if stemmed in feature_names:
            idx = np.where(feature_names == stemmed)[0][0]
            weight = feature_weights[idx]
            if abs(weight) > 0.1: # Significant contribution
                explanations.append({
                    "word": word,
                    "weight": float(weight),
                    "impact": "Spam" if weight > 0 else "Ham"
                })
                
    # Sort by absolute weight
    explanations.sort(key=lambda x: abs(x['weight']), reverse=True)
    return explanations[:10] # Top 10 words
