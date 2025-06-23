import os
import requests
from fastapi import FastAPI
from siamese_app import router as siamese_router
from ocr_app import router as ocr_router

# Constants
MODEL_URL = "https://huggingface.co/z1th1z/RoofType_Detect/resolve/main/siamese_model.pth"
MODEL_PATH = "siamese_model.pth"

# Model downloader
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading Siamese model from Hugging Face...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("✅ Model download complete.")

# Ensure model is available before app starts
download_model()

# FastAPI app setup
app = FastAPI(
    title="Unified ML API",
    description="Combines OCR and Siamese model inference.",
    version="1.0.0"
)

# Routers
app.include_router(siamese_router, prefix="/siamese", tags=["Siamese Model"])
app.include_router(ocr_router, prefix="/ocr", tags=["OCR"])
