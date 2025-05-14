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
    description="Dynamic ML model serving via FastAPI on Cloud Run",
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

def get_model_type_from_metadata() -> str:
    """Extract model type from metadata with fallback"""
    if model_metadata:
        # Try multiple possible keys for model type
        model_type = (
            model_metadata.get('best_model_name') or 
            model_metadata.get('model_type') or 
            model_metadata.get('algorithm')
        )
        if model_type:
            return str(model_type)
    
    # Fallback: try to detect from the model object itself
    if model:
        model_class = model.__class__.__name__
        if hasattr(model, 'named_steps'):  # Pipeline
            # Get the actual model from pipeline
            for step_name, step in model.named_steps.items():
                if 'model' in step_name.lower():
                    model_class = step.__class__.__name__
                    break
        
        # Map class names to readable names
        class_mapping = {
            'LGBMRegressor': 'LightGBM',
            'XGBRegressor': 'XGBoost',
            'RandomForestRegressor': 'Random Forest',
            'ExtraTreesRegressor': 'Extra Trees',
            'LinearRegression': 'Linear Regression',
            'Ridge': 'Ridge Regression',
            'Lasso': 'Lasso Regression',
            'SVR': 'Support Vector Regression'
        }
        
        return class_mapping.get(model_class, model_class)
    
    return 'Unknown'

def get_model_version() -> str:
    """Extract model version/timestamp from metadata"""
    if model_metadata:
        return (
            model_metadata.get('timestamp') or 
            model_metadata.get('version') or 
            'Unknown'
        )
    return 'Unknown'

def get_model_performance() -> Dict[str, Any]:
    """Extract model performance metrics from metadata"""
    if model_metadata and 'performance' in model_metadata:
        return model_metadata['performance']
    return {}

@app.on_event("startup")
async def startup_event():
    """Load model on startup with detailed logging"""
    logger.info("🚀 Starting up ML Prediction Service...")
    logger.info(f"MODEL_BUCKET environment variable: {os.getenv('MODEL_BUCKET', 'NOT SET')}")
    
    try:
        await load_latest_model()
        logger.info("✅ Model loaded successfully during startup")
    except Exception as e:
        logger.error(f"❌ Failed to load model during startup: {e}")
        logger.info("⚠️ Service will continue without model. Use /model/reload to load manually.")

async def load_latest_model():
    """Load the latest model from Cloud Storage with detailed logging"""
    global model, model_metadata
    
    bucket_name = os.getenv('MODEL_BUCKET', 'ml-data-451319')
    logger.info(f"🔍 Looking for models in bucket: {bucket_name}")
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        logger.info(f"✅ Connected to bucket: {bucket_name}")
        
        # Find latest model with more detailed logging
        model_files = []
        logger.info("🔍 Scanning for model files...")
        
        blobs = list(bucket.list_blobs(prefix='models/'))
        logger.info(f"Found {len(blobs)} objects in models/ folder")
        
        for blob in blobs:
            logger.info(f"Checking blob: {blob.name}")
            if blob.name.endswith('_model.pkl'):
                timestamp = blob.name.split('/')[-1].replace('_model.pkl', '')
                model_files.append((timestamp, blob.name))
                logger.info(f"Found model file: {blob.name} (timestamp: {timestamp})")
        
        if not model_files:
            raise Exception(f"No model files found in bucket {bucket_name}/models/")
        
        # Get latest model
        model_files.sort(reverse=True)
        latest_timestamp, latest_model_path = model_files[0]
        logger.info(f"🎯 Selected latest model: {latest_model_path}")
        
        # Download model
        temp_model_path = "/tmp/model.pkl"
        model_blob = bucket.blob(latest_model_path)
        
        logger.info(f"⬇️ Downloading model from {latest_model_path}...")
        model_blob.download_to_filename(temp_model_path)
        logger.info(f"✅ Model downloaded to {temp_model_path}")
        
        # Load model
        logger.info("📦 Loading model with joblib...")
        model = joblib.load(temp_model_path)
        logger.info(f"✅ Model loaded successfully! Type: {type(model)}")
        
        # Try to load metadata
        try:
            metadata_path = latest_model_path.replace('_model.pkl', '_metadata.json')
            logger.info(f"🔍 Looking for metadata at: {metadata_path}")
            metadata_blob = bucket.blob(metadata_path)
            metadata_text = metadata_blob.download_as_text()
            model_metadata = json.loads(metadata_text)
            logger.info("✅ Model metadata loaded")
            logger.info(f"📊 Model type from metadata: {get_model_type_from_metadata()}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load metadata: {e}")
            model_metadata = {"version": latest_timestamp}
        
        # Clean up temp file
        os.remove(temp_model_path)
        logger.info("🧹 Cleaned up temporary model file")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": get_model_type_from_metadata() if model else None,
        "service": "ML Prediction Service",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_type": get_model_type_from_metadata(),
        "model_version": get_model_version()
    }

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
                "model_type": get_model_type_from_metadata(),
                "version": get_model_version(),
                "prediction_count": 1,
                "performance": get_model_performance()
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
                "model_type": get_model_type_from_metadata(),
                "version": get_model_version(),
                "prediction_count": len(predictions),
                "performance": get_model_performance()
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
                "model_type": get_model_type_from_metadata(),
                "version": get_model_version(),
                "prediction_count": len(predictions),
                "performance": get_model_performance()
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
        "model_type": get_model_type_from_metadata(),
        "version": get_model_version(),
        "feature_count": len(feature_names),
        "features": feature_names,
        "performance": get_model_performance()
    }
    
    # Add feature importance if available
    if model_metadata and 'feature_importance' in model_metadata:
        response["feature_importance"] = model_metadata['feature_importance']
    
    # Add all model comparison data if available
    if model_metadata and 'all_model_comparison' in model_metadata:
        response["all_model_comparison"] = model_metadata['all_model_comparison']
    
    return response

@app.post("/model/reload")
async def reload_model():
    """Manually reload the latest model"""
    try:
        await load_latest_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_type": get_model_type_from_metadata(),
            "model_version": get_model_version(),
            "model_loaded": model is not None
        }
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        return {
            "status": "error", 
            "message": f"Model reload failed: {str(e)}",
            "model_loaded": model is not None
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)