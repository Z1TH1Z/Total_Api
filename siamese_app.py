import os
import io
import torch
from torchvision import transforms
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from one_shot_model import Siamese

router = APIRouter()

MODEL_PATH = "siamese_model.pth"
REFERENCE_DIR = "training"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((105, 105)),
    transforms.ToTensor()
])

# Class label → reference image paths
class_to_images = {}

# Lazy-loaded model
model = None

def get_model():
    """Loads and returns the Siamese model only when needed."""
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file {MODEL_PATH} not found.")
        print("🔁 Loading Siamese model...")
        m = Siamese()
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.eval()
        m.to(DEVICE)
        model = m
    return model

def load_reference_images(reference_dir):
    """Scans training dir and maps class labels to image file paths."""
    global class_to_images
    class_to_images = {}
    if not os.path.isdir(reference_dir):
        return
    for class_name in os.listdir(reference_dir):
        class_path = os.path.join(reference_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        images = [os.path.join(class_path, f) for f in os.listdir(class_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]
        if images:
            class_to_images[class_name] = images

# Load once at startup
load_reference_images(REFERENCE_DIR)

@router.get("/")
async def root():
    return {"message": "Siamese model inference API ready. POST to /predict/image."}

@router.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    """Predicts roof type by comparing input image to reference images."""
    if not class_to_images:
        raise HTTPException(status_code=503, detail="No reference images loaded.")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        model = get_model()  # Lazy-load here

        scores = {}
        with torch.no_grad():
            for class_name, ref_paths in class_to_images.items():
                class_scores = []
                for ref_path in ref_paths:
                    ref_img = Image.open(ref_path).convert('L')
                    ref_img_tensor = transform(ref_img).unsqueeze(0).to(DEVICE)
                    output = model(img_tensor, ref_img_tensor)
                    prob = torch.sigmoid(output).item()
                    class_scores.append(prob)
                scores[class_name] = sum(class_scores) / len(class_scores)

        predicted_class = max(scores, key=scores.get)
        return JSONResponse(content={
            "predicted_class": predicted_class,
            "similarity_scores": {cls: round(score, 4) for cls, score in scores.items()}
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@router.post("/reload_references/")
async def reload_references():
    try:
        load_reference_images(REFERENCE_DIR)
        return JSONResponse(content={"message": "Reference images reloaded."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload: {e}")
