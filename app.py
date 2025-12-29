from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Load model and artifacts
model = load_model('models/sentiment_model.h5')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('models/config.pkl', 'rb') as f:
    config = pickle.load(f)
final_recommendations = pd.read_csv('models/recommendations.csv')
brand_affinity = pd.read_csv('models/brand_affinity.csv')
category_affinity = pd.read_csv('models/category_affinity.csv')
product_metadata = pd.read_csv('models/product_metadata.csv')

class ReviewRequest(BaseModel):
    text: str

class RecommendationRequest(BaseModel):
    username: str
    top_k: int = 5

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API", "status": "running"}

@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    # Tokenize and pad (same as training)
    sequence = tokenizer.texts_to_sequences([request.text])
    padded = pad_sequences(sequence, maxlen=config['max_length'], padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    sentiment_idx = np.argmax(prediction[0])
    reverse_label_map = {v: k for k, v in config['label_map'].items()}
    sentiment = reverse_label_map[sentiment_idx]
    
    return {
        "sentiment": sentiment,
        "confidence": float(prediction[0][sentiment_idx]),
        "probabilities": {
            "negative": float(prediction[0][0]),
            "neutral": float(prediction[0][1]),
            "positive": float(prediction[0][2])
        }
    }

@app.post("/recommend")
def recommend_products(request: RecommendationRequest):
    # EXACT SAME LOGIC AS YOUR recommend_for_user FUNCTION
    user = request.username
    top_k = request.top_k
    
    brand_aff = brand_affinity[brand_affinity["reviews.username"] == user]
    cat_aff = category_affinity[category_affinity["reviews.username"] == user]
    
    candidates = product_metadata.merge(brand_aff, on="brand", how="left") \
                             .merge(cat_aff, on="primary_category", how="left")
    
    candidates["brand_affinity"].fillna(0, inplace=True)
    candidates["category_affinity"].fillna(0, inplace=True)
    
    candidates["affinity_score"] = (
        0.6 * candidates["brand_affinity"] +
        0.4 * candidates["category_affinity"]
    )
    
    candidates = candidates.merge(
        final_recommendations[["name", "final_score"]],
        on="name",
        how="left"
    )
    
    candidates["sentiment_score"] = candidates["final_score"].fillna(0)
    candidates["final_score"] = (
        0.7 * candidates["sentiment_score"] +
        0.3 * candidates["affinity_score"]
    )
    
    candidates.drop(columns=["brand_affinity", "category_affinity", "reviews.username_x", "reviews.username_y"], inplace=True)
    
    top_products = candidates.sort_values("final_score", ascending=False).head(top_k)
    
    return {
        "username": user,
        "recommendations": top_products[["name", "brand", "primary_category", "final_score"]].to_dict('records')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)