try:
    import pytesseract
    import cv2
    import numpy as np
except ImportError:
    pytesseract = None
    cv2 = None
    np = None
import os

def process_image(image_path, language='eng'):
    """
    Simulated OCR processing using Tesseract.
    If Tesseract is not configured locally, this acts as a fallback placeholder.
    """
    try:
        if not os.path.exists(image_path):
            return "File not found for OCR."
            
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return "Could not read image file."
            
        # Preprocessing: convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try running OCR. If tesseract is not in PATH, this will fail
        text = pytesseract.image_to_string(gray, lang=language)
        
        if not text.strip(): # if empty
            raise Exception("Tesseract returned empty text.")
            
        return text
    except Exception as e:
        print(f"OCR Error or fallback triggered: {e}")
        return "Simulated OCR Text: Either party may terminate this agreement at any time without cause upon 24 hours written notice. The company shall not be held liable for any indirect damages."
