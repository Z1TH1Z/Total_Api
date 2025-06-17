from fastapi import APIRouter, File, UploadFile
from ocr_test import extract_meter_info

router = APIRouter()

@router.post("/extract-info/")
async def extract_info(file: UploadFile = File(...)):
    contents = await file.read()
    info = extract_meter_info(contents)
    return {"filename": file.filename, "extracted_info": info}
