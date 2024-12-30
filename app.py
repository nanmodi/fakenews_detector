from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model package with safer path handling
model_path = r'best_fake_news_detector.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

try:
    model_package = joblib.load(model_path)

    model = model_package['model']
    print(model)
    vectorizer = model_package['vectorizer']
except KeyError as e:
    raise HTTPException(status_code=500, detail=f"Missing key in model package: {str(e)}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

class NewsText(BaseModel):
    text: str

class Prediction(BaseModel):
    is_fake: bool
    confidence: float

@app.post("/api/predict", response_model=Prediction)
async def predict_news(news: NewsText):
    try:
        # Ensure input text is valid
        if not news.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Transform the input text using the vectorizer
        X = vectorizer.transform([news.text])
        
        # Get prediction and probability (if available)
        prediction = model.predict(X)[0]
        
        # For models like SVM, we don't have predict_proba, so we use decision_function for confidence
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0]
            confidence = max(prob)
            print(confidence)
        else:
            # For SVM or models that don't support predict_proba, use decision function to estimate confidence
            decision = model.decision_function(X)[0]
            
            confidence = 1 / (1 + np.exp(-abs(decision))) 
            print(confidence) # Sigmoid function for probability-like value
        
        

        return Prediction(
            is_fake=bool(prediction),
            confidence=float(confidence))
    except Exception as e:
        # Return a detailed error message to the client if any issue occurs
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=3000)
