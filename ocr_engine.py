import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import os

class OCREngine:
    def __init__(self, languages=None):
        self.languages = languages or ["eng"]
        self.lang_string = "+".join(self.languages)
        
        # Configure tesseract if needed
        # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Adjust path if needed
    
    def extract_text_from_image_data(self, image_data):
        """Extract text from image data using OCR"""
        
        try:
            # Convert image data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Preprocess image for better OCR
            processed_img = self.preprocess_image(img_array)
            
            # Extract text using tesseract
            custom_config = f'--oem 3 --psm 6 -l {self.lang_string}'
            text = pytesseract.image_to_string(processed_img, config=custom_config)
            
            # Clean and format text
            cleaned_text = self.clean_text(text)
            
            return cleaned_text
            
        except Exception as e:
            print(f"Error in OCR processing: {str(e)}")
            return ""
    
    def preprocess_image(self, img_array):
        """Preprocess image for better OCR results"""
        
        try:
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply threshold to get binary image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            print(f"Error in image preprocessing: {str(e)}")
            return img_array
    
    def clean_text(self, text):
        """Clean and format extracted text"""
        
        if not text:
            return ""
        
        # Remove extra whitespaces and normalize
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        # Join lines with proper spacing
        cleaned_text = '\n'.join(cleaned_lines)
        
        return cleaned_text
    
    def extract_text_with_confidence(self, image_data, min_confidence=60):
        """Extract text with confidence scores"""
        
        try:
            image = Image.open(io.BytesIO(image_data))
            img_array = np.array(image)
            processed_img = self.preprocess_image(img_array)
            
            # Get detailed OCR data
            custom_config = f'--oem 3 --psm 6 -l {self.lang_string}'
            data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter by confidence
            filtered_text = []
            for i, confidence in enumerate(data['conf']):
                if int(confidence) > min_confidence:
                    text = data['text'][i].strip()
                    if text:
                        filtered_text.append(text)
            
            return ' '.join(filtered_text)
            
        except Exception as e:
            print(f"Error in confidence-based OCR: {str(e)}")
            return self.extract_text_from_image_data(image_data)
