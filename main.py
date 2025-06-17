from fastapi import FastAPI
from siamese_app import router as siamese_router
from ocr_app import router as ocr_router

app = FastAPI(
    title="Unified ML API",
    description="Combines OCR and Siamese model inference.",
    version="1.0.0"
)

app.include_router(siamese_router, prefix="/siamese", tags=["Siamese Model"])
app.include_router(ocr_router, prefix="/ocr", tags=["OCR"])
