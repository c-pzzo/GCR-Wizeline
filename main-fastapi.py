from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import os
import io
import json
import asyncio
from google.cloud import storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Prediction Service",
    description="LightGBM model serving via FastAPI on Cloud Run",
    version="1.0.0"
)

# Global model variable
model = None
model_metadata = None
feature_names = [f'feature_{i}' for i in range(20)]

# Pydantic models for request/response validation
class SinglePredictionRequest(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float

class BatchPredictionRequest(BaseModel):
    instances: List[SinglePredictionRequest]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Load model on startup - with timeout handling"""
    try:
        # Load model asynchronously with timeout
        task = asyncio.create_task(load_latest_model())
        await asyncio.wait_for(task, timeout=60.0)  # 60 second timeout
    except asyncio.TimeoutError:
        logger.warning("Model loading timed out during startup. Will load on first request.")
    except Exception as e:
        logger.warning(f"Failed to load model during startup: {e}. Will retry on first request.")


async def load_latest_model():
    """Load the latest model from Cloud Storage"""
    global model, model_metadata
    
    try:
        bucket_name = os.getenv('MODEL_BUCKET', 'ml-data-451319')
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Find latest model
        model_files = []
        for blob in bucket.list_blobs(prefix='models/'):
            if blob.name.endswith('_model.pkl'):
                timestamp = blob.name.split('/')[-1].replace('_model.pkl', '')
                model_files.append((timestamp, blob.name))
        
        if not model_files:
            raise Exception("No model files found!")
        
        # Get latest model
        model_files.sort(reverse=True)
        latest_timestamp, latest_model_path = model_files[0]
        
        # Download model
        temp_model_path = "/tmp/model.pkl"
        model_blob = bucket.blob(latest_model_path)
        model_blob.download_to_filename(temp_model_path)
        
        # Load model
        model = joblib.load(temp_model_path)
        logger.info(f"✅ Model loaded successfully: {latest_timestamp}")
        
        # Try to load metadata
        try:
            metadata_path = latest_model_path.replace('_model.pkl', '_metadata.json')
            metadata_blob = bucket.blob(metadata_path)
            metadata_text = metadata_blob.download_as_text()
            model_metadata = json.loads(metadata_text)
            logger.info("✅ Model metadata loaded")
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            model_metadata = {"version": latest_timestamp}
        
        # Clean up temp file
        os.remove(temp_model_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "service": "ML Prediction Service",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": model_metadata.get('best_model_name', 'Unknown') if model_metadata else 'Unknown',
        "timestamp": model_metadata.get('timestamp', 'Unknown') if model_metadata else 'Unknown'
    }

async def ensure_model_loaded():
    """Ensure model is loaded before prediction"""
    global model
    if model is None:
        await load_latest_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: SinglePredictionRequest):
    """Single prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to DataFrame
        data = pd.DataFrame([request.dict()])
        
        # Ensure correct feature order
        data = data.reindex(columns=feature_names)
        
        # Make prediction
        prediction = model.predict(data.values)
        
        return PredictionResponse(
            predictions=prediction.tolist(),
            model_info={
                "model_type": model_metadata.get('best_model_name', 'LightGBM') if model_metadata else 'LightGBM',
                "version": model_metadata.get('timestamp', 'Unknown') if model_metadata else 'Unknown',
                "prediction_count": 1
            }
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=PredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert instances to DataFrame
        data_list = [instance.dict() for instance in request.instances]
        data = pd.DataFrame(data_list)
        
        # Ensure correct feature order
        data = data.reindex(columns=feature_names)
        
        # Make predictions
        predictions = model.predict(data.values)
        
        return PredictionResponse(
            predictions=predictions.tolist(),
            model_info={
                "model_type": model_metadata.get('best_model_name', 'LightGBM') if model_metadata else 'LightGBM',
                "version": model_metadata.get('timestamp', 'Unknown') if model_metadata else 'Unknown',
                "prediction_count": len(predictions)
            }
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...)):
    """CSV file prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Check if required features exist
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing features: {missing_features}"
            )
        
        # Select and order features
        df = df[feature_names]
        
        # Make predictions
        predictions = model.predict(df.values)
        
        # Create response
        response_data = {
            "predictions": predictions.tolist(),
            "row_count": len(predictions),
            "model_info": {
                "model_type": model_metadata.get('best_model_name', 'LightGBM') if model_metadata else 'LightGBM',
                "version": model_metadata.get('timestamp', 'Unknown') if model_metadata else 'Unknown',
                "prediction_count": len(predictions)
            }
        }
        
        return response_data
        
    except Exception as e:
        logger.error(f"CSV prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"CSV prediction failed: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get current model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    response = {
        "model_loaded": True,
        "feature_count": len(feature_names),
        "features": feature_names
    }
    
    if model_metadata:
        response.update({
            "model_type": model_metadata.get('best_model_name', 'Unknown'),
            "timestamp": model_metadata.get('timestamp', 'Unknown'),
            "performance": model_metadata.get('performance', {}),
            "feature_importance": model_metadata.get('feature_importance', [])
        })
    
    return response

@app.post("/model/reload")
async def reload_model():
    """Manually reload the latest model"""
    try:
        await load_latest_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_type": model_metadata.get('best_model_name', 'Unknown') if model_metadata else 'Unknown'
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)