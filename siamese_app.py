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

model = Siamese()
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    model.to(DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

class_to_images = {}

def load_reference_images(reference_dir):
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

load_reference_images(REFERENCE_DIR)

@router.get("/")
async def read_root():
    return {"message": "Welcome to the Siamese Network Inference API. Use /predict/image to classify an image."}

@router.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    if not class_to_images:
        raise HTTPException(
            status_code=503,
            detail="Reference images not loaded."
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('L')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

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
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@router.post("/reload_references/")
async def reload_references():
    try:
        load_reference_images(REFERENCE_DIR)
        return JSONResponse(content={"message": "Reference images reloaded."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload references: {e}")
