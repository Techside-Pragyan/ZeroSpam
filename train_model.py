import pandas as pd
import numpy as np
import os
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download NLTK data
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
    return " ".join(cleaned_tokens)

def train_model():
    data_path = "data/spam_dataset.csv"
    if not os.path.exists(data_path):
        print("Dataset not found. Please run download_data.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Filter out empty texts after cleaning
    df = df[df['cleaned_text'].str.strip() != '']
    
    X = df['cleaned_text']
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Extracting features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training model...")
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train_tfidf, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))
    
    # Save model and vectorizer
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/spam_classifier.joblib')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    print("Model and vectorizer saved to 'models/' directory.")

if __name__ == "__main__":
    train_model()
