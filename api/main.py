from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import sys
from typing import List, Optional

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.nlp_utils import clean_text, explain_prediction

app = FastAPI(title="ZeroSpam API", description="AI-powered Email Spam Detection")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
MODEL_PATH = "models/spam_classifier.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    is_spam: bool
    explanation: List[dict]

@app.get("/")
async def root():
    return {"message": "ZeroSpam API is running", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    
    cleaned = " ".join(clean_text(request.text))
    if not cleaned:
         return {
            "label": "Ham",
            "confidence": 1.0,
            "is_spam": False,
            "explanation": []
        }

    # Vectorize
    tfidf_vec = vectorizer.transform([cleaned])
    
    # Predict
    prob = model.predict_proba(tfidf_vec)[0]
    is_spam = bool(model.predict(tfidf_vec)[0])
    
    confidence = float(prob[1] if is_spam else prob[0])
    label = "Spam" if is_spam else "Ham"
    
    # Explain
    explanation = explain_prediction(request.text, model, vectorizer)
    
    return {
        "label": label,
        "confidence": confidence,
        "is_spam": is_spam,
        "explanation": explanation
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(('.txt', '.eml')):
        raise HTTPException(status_code=400, detail="Only .txt and .eml files are supported.")
    
    content = await file.read()
    text = content.decode("utf-8", errors="ignore")
    
    # In a real EML parser, we'd extract subject and body. For now, we'll treat it as text.
    return await predict(PredictionRequest(text=text))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
