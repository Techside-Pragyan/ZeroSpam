# ZeroSpam AI | Spam Email Classification System

ZeroSpam is a full-stack, AI-powered system designed to classify emails and messages as **Spam** or **Ham** (Not Spam) with high accuracy and explainability.

## 🚀 Key Features

- **Real-time Classification**: Instant analysis of text content.
- **Explainability Report**: Highlights specific words and patterns that triggered the spam detection.
- **File Support**: Upload `.txt` or `.eml` files for batch or single analysis.
- **Advanced NLP**: Uses TF-IDF vectorization and Logistic Regression for high-performance inference.
- **Premium UI**: Modern glassmorphism design with responsive layouts.

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: React, Vite, Framer Motion
- **ML/AI**: Scikit-Learn, NLTK
- **Icons**: Lucide React

## 📂 Project Structure

- `/api`: FastAPI backend and endpoints.
- `/frontend`: React application.
- `/models`: Trained ML models and vectorizers.
- `/utils`: NLP preprocessing and explainability logic.
- `/data`: Dataset management and download scripts.

## 🏁 Getting Started

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download and train the model
python data/download_data.py
python train_model.py

# Start the API server
uvicorn api.main:app --reload
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🧠 Explainability Note
ZeroSpam provides a weight-based explanation for each classification, showing exactly how much each word influenced the "Spam" vs "Ham" decision.