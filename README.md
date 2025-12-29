# Sentiment-Aware-Product-Recommendation-System
An end-to-end deep learning system that analyzes customer reviews to predict sentiment and enhance product recommendations. The pipeline includes text preprocessing, sentiment classification using deep learning models, brand and user preference profiling, and sentiment-aware ranking to deliver personalized recommendations.
Note: For detailed explanation of the python code for modeling the recommendation system refer: [codefile.ipynb]
# Sentiment Analysis - Complete Docker Guide

## 📁 Project Structure

```
sentiment-project/
├── Dockerfile
├── requirements.txt
├── train.py
├── app.py
├── data/
│   └── GrammarandProductReviews.csv
└── models/              (created after training)
    ├── sentiment_model.h5
    ├── tokenizer.pkl
    ├── config.pkl
    └── *.csv files
```

---

## Step-by-Step Execution Guide

### Prerequisites

✅ Docker Desktop installed and running
✅ All files created in project folder
✅ CSV file placed in `data/` folder

---

## Step 1: Create Project Structure

### 1.1 Create Project Folder
```powershell
# Create and navigate to project folder
mkdir sentiment-project
cd sentiment-project
```

### 1.2 Create Required Folders
```powershell
mkdir data
mkdir models
```

### 1.3 Create the Dockerfile and requirements.txt 
```powershell
Both the Dockerfile and requirements.txt are uploaded.
```

### 1.4 Add Your Files
- Copy `train.py` to project folder
- Copy `app.py` to project folder
- Copy `GrammarandProductReviews.csv` to `data/` folder

---

## Step 2: Build Docker Image

### 2.1 Start Docker Desktop
- Open Docker Desktop application
- Wait until whale icon is steady 

### 2.2 Verify Docker is Running
```powershell
docker --version
```

**Expected output:**
```
Docker version 24.0.x, build xxxxx
```

### 2.3 Build the Image
```powershell
docker build -t sentiment-model .
```

**What to expect:**
- This takes 5-15 minutes (first time)
- You'll see multiple steps executing
- Downloads Python, installs packages
- Final message: "Successfully tagged sentiment-model:latest"

**✅ Validation:**
```powershell
docker images
```

**Expected output:**
```
REPOSITORY         TAG       IMAGE ID       CREATED         SIZE
sentiment-model    latest    abc123def456   1 minute ago    2.5GB
```

---

## Step 3: Train the Model

### 3.1 Run Training
```powershell
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data sentiment-model python train.py
```
- Takes 10-30 minutes depending on data size

**✅ Validation:**
```powershell
# Check if model files were created
Get-ChildItem models/
```

**Expected output:**
```
sentiment_model.h5
tokenizer.pkl
config.pkl
recommendations.csv
brand_affinity.csv
category_affinity.csv
product_metadata.csv
```

**Check file sizes:**
```powershell
Get-ChildItem models/ | Select-Object Name, Length
```

**Expected sizes:**
- `sentiment_model.h5`: ~10-50 MB
- `tokenizer.pkl`: ~1-5 MB
- `config.pkl`: ~1 KB
- CSV files: Varies based on data

**⚠️ If models folder is empty:**
- Training failed - check error messages
- Make sure CSV file is in `data/` folder
- Re-run training command

---

## Step 4: Deploy API

### 4.1 Start API Server
```powershell
docker run -p 8000:8000 -v ${PWD}/models:/app/models:ro sentiment-model python app.py
```

**API is now running!**
Keep this PowerShell window open. API runs until you press `Ctrl+C`.

---

## Step 5: Validate & Test API

### 5.1 Check API Health

**Open new PowerShell window** (keeping API running in first window)
```powershell
curl http://localhost:8000/
```

**Expected output:**
```json
{"message":"Sentiment Analysis API","status":"running"}
```

### 5.2 Test Sentiment Prediction

```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "Content-Type: application/json" `
  -d '{"text": "This product is absolutely amazing! I love it so much."}'
```

**Expected output:**
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.02,
    "neutral": 0.03,
    "positive": 0.95
  }
}
```

**✅ Validation Checks:**
- ✅ Sentiment should be "positive" for positive text
- ✅ Confidence should be > 0.7 for clear sentiment
- ✅ Probabilities should sum to ~1.0


### 5.3 Test Recommendations

```powershell
curl -X POST "http://localhost:8000/recommend" `
  -H "Content-Type: application/json" `
  -d '{"username": "An anonymous customer", "top_k": 5}'
```

**Expected output:**
```json
{
  "username": "An anonymous customer",
  "recommendations": [
    {
      "name": "Product Name 1",
      "brand": "Brand A",
      "primary_category": "Electronics",
      "final_score": 4.8
    },
    {
      "name": "Product Name 2",
      "brand": "Brand B",
      "primary_category": "Home",
      "final_score": 4.5
    },
    ...
  ]
}
```

**✅ Validation Checks:**
- ✅ Returns 5 products (or fewer if less available)
- ✅ Products have scores between 0-5
- ✅ Products are sorted by final_score (highest first)

### 5.4 Interactive API Documentation

**Open browser and visit:**
```
http://localhost:8000/docs
```

**What you'll see:**
- Interactive UI
- All available endpoints
- "Try it out" buttons to test directly in browser

**✅ Validation:**
1. Click on `/predict` endpoint
2. Click "Try it out"
3. Enter sample text
4. Click "Execute"
5. See prediction response below

---

## Step 6: Validate Results Quality

Look at training output for accuracy metrics:
```
Test accuracy: 0.85-0.90  ← Should be above 0.80
```

If accuracy is below 0.80:
- Model needs more training data
- Consider adjusting hyperparameters
- Check data quality

---

## Step 7: Stop and Clean Up

### 7.1 Stop API Server
In the PowerShell window running the API:
```
Press Ctrl+C
```

**Expected output:**
```
INFO:     Shutting down
INFO:     Finished server shutdown.
```

### 7.2 Check Running Containers
```powershell
docker ps
```

If any containers are still running:
```powershell
docker stop <container-id>
```

### 7.3 View All Containers (including stopped)
```powershell
docker ps -a
```

### 7.4 Remove Stopped Containers (Optional)
```powershell
docker container prune
```

## Troubleshooting Guide

### Issue: "Docker Desktop is not running"
**Solution:**
- Open Docker Desktop
- Wait for whale icon to be steady
- Run `docker ps` to verify

### Issue: "No such file or directory: data/GrammarandProductReviews.csv"
**Solution:**
```powershell
# Verify CSV location
Test-Path data/GrammarandProductReviews.csv
# Should return True
```

### Issue: "Models folder is empty after training"
**Solution:**
- Check training output for errors
- Re-run training with verbose output
- Ensure CSV file has data

### Issue: "Port 8000 already in use"
**Solution:**
```powershell
# Find and stop process using port 8000
netstat -ano | findstr :8000
# Note the PID, then:
taskkill /PID <PID> /F
```

### Issue: "API returns 500 error"
**Solution:**
- Ensure training completed successfully
- Check that all 7 model files exist
- Restart API container

### Issue: "Predictions seem random"
**Solution:**
- Check training accuracy (should be >0.80)
- Ensure enough training data (>1000 reviews)
- Retrain with more epochs

### Issue: "Build takes too long"
**Solution:**
- First build takes 10-15 minutes (normal)
- Subsequent builds use cache (faster)
- Check internet connection

---

## Quick Command Reference

```powershell
# Build image
docker build -t sentiment-model .

# Train model
docker run -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data sentiment-model python train.py

# Start API
docker run -p 8000:8000 -v ${PWD}/models:/app/models:ro sentiment-model python app.py

# Test prediction
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"text": "Great product!"}'

# Test recommendations
curl -X POST "http://localhost:8000/recommend" -H "Content-Type: application/json" -d '{"username": "testuser", "top_k": 5}'

# View running containers
docker ps

# Stop container
docker stop <container-id>

# View logs
docker logs <container-id>
```

## Success Criteria

Your setup is successful if:

✅ Docker image builds without errors
✅ Training completes with >0.80 accuracy
✅ 7 model files created in `models/` folder
✅ API starts and shows "Uvicorn running"
✅ Health check returns `{"status":"running"}`
✅ Positive reviews predict as "positive"
✅ Negative reviews predict as "negative"
✅ Recommendations return products with scores
✅ Interactive docs work at `/docs`

**Congratulations! Your sentiment analysis API is now running!** 🚀

---

Note: This ReadMe file was enhanced by the use of AI.
Note: Dataset is not uploaded to ensure the privacy.

Need help? Check the troubleshooting section or review error messages carefully!
