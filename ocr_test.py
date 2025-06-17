import easyocr
import cv2
import numpy as np
import re

reader = easyocr.Reader(['en'])

def extract_meter_info(image_bytes):
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = reader.readtext(image)

    extracted_info = {
        "kh": None,
        "frequency": None,
        "voltage": None,
        "serial_number": None,
        "other_specs": []
    }

    kh_pattern = re.compile(r"\bKh[:\s]*([0-9.]+)", re.IGNORECASE)
    freq_pattern = re.compile(r"\b([0-9]+)\s*Hz\b", re.IGNORECASE)
    volt_pattern = re.compile(r"\b([0-9]{3})\s*v\b", re.IGNORECASE)
    serial_pattern = re.compile(r"\b(\d{7,})\b")

    for (_, text, _) in results:
        text_clean = text.strip()

        if kh_match := kh_pattern.search(text_clean):
            extracted_info["kh"] = kh_match.group(1)
        elif freq_match := freq_pattern.search(text_clean):
            extracted_info["frequency"] = freq_match.group(1)
        elif volt_match := volt_pattern.search(text_clean):
            extracted_info["voltage"] = volt_match.group(1)
        elif serial_match := serial_pattern.search(text_clean):
            if not extracted_info["serial_number"] and not re.search(r"hz|v", text_clean.lower()):
                extracted_info["serial_number"] = serial_match.group(1)
        else:
            extracted_info["other_specs"].append(text_clean)

    return extracted_info
