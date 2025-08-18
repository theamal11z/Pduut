import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import pytesseract

class AdvancedOCRProcessor:
    """Advanced OCR preprocessing for better text extraction"""
    
    def __init__(self):
        self.preprocessing_methods = [
            'gaussian_blur',
            'median_filter', 
            'morphological_operations',
            'contrast_enhancement',
            'noise_reduction',
            'skew_correction'
        ]
    
    def preprocess_image_advanced(self, image_data, method='auto'):
        """Apply advanced preprocessing techniques for better OCR"""
        
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array
            img_array = np.array(image)
            
            if method == 'auto':
                # Automatically select best preprocessing
                processed_img = self.auto_preprocess(img_array)
            else:
                processed_img = self.apply_specific_preprocessing(img_array, method)
            
            return processed_img
            
        except Exception as e:
            print(f"Error in advanced preprocessing: {str(e)}")
            return np.array(Image.open(io.BytesIO(image_data)))
    
    def auto_preprocess(self, img_array):
        """Automatically determine and apply best preprocessing"""
        
        # Analyze image characteristics
        analysis = self.analyze_image_quality(img_array)
        
        # Apply preprocessing based on analysis
        processed = img_array.copy()
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        
        # Apply techniques based on image quality
        if analysis['needs_denoising']:
            processed = self.reduce_noise(processed)
        
        if analysis['needs_contrast_enhancement']:
            processed = self.enhance_contrast(processed)
        
        if analysis['needs_sharpening']:
            processed = self.apply_sharpening(processed)
            
        if analysis['has_skew']:
            processed = self.correct_skew(processed)
        
        # Final binarization
        processed = self.adaptive_binarization(processed)
        
        return processed
    
    def analyze_image_quality(self, img_array):
        """Analyze image to determine preprocessing needs"""
        
        analysis = {
            'needs_denoising': False,
            'needs_contrast_enhancement': False,
            'needs_sharpening': False,
            'has_skew': False,
            'quality_score': 0.5
        }
        
        try:
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Check contrast
            contrast = np.std(gray)
            if contrast < 50:
                analysis['needs_contrast_enhancement'] = True
            
            # Check for noise (using local variance)
            kernel = np.ones((3,3), np.uint8)
            local_variance = cv2.filter2D(gray.astype(np.float32), -1, kernel/9)
            noise_level = np.mean(np.abs(local_variance - gray.astype(np.float32)))
            
            if noise_level > 30:
                analysis['needs_denoising'] = True
            
            # Check sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 500:
                analysis['needs_sharpening'] = True
            
            # Simple skew detection using Hough lines
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    angles.append(angle)
                
                if len(angles) > 0:
                    mean_angle = np.mean(angles)
                    if abs(mean_angle - 90) > 2:  # More than 2 degrees skew
                        analysis['has_skew'] = True
            
            # Calculate overall quality score
            quality_factors = [
                min(1.0, contrast / 100),  # Contrast factor
                max(0, 1.0 - noise_level / 100),  # Noise factor
                min(1.0, laplacian_var / 1000)  # Sharpness factor
            ]
            analysis['quality_score'] = np.mean(quality_factors)
            
        except Exception as e:
            print(f"Error analyzing image quality: {str(e)}")
        
        return analysis
    
    def reduce_noise(self, image):
        """Apply noise reduction techniques"""
        
        try:
            # Median filter for salt-and-pepper noise
            denoised = cv2.medianBlur(image, 3)
            
            # Gaussian blur for general noise
            denoised = cv2.GaussianBlur(denoised, (3, 3), 0)
            
            # Non-local means denoising for better quality
            denoised = cv2.fastNlMeansDenoising(denoised, None, 10, 7, 21)
            
            return denoised
            
        except Exception as e:
            print(f"Error in noise reduction: {str(e)}")
            return image
    
    def enhance_contrast(self, image):
        """Enhance image contrast using CLAHE"""
        
        try:
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            
            # Apply CLAHE
            enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            print(f"Error in contrast enhancement: {str(e)}")
            return image
    
    def apply_sharpening(self, image):
        """Apply sharpening filter to improve text clarity"""
        
        try:
            # Sharpening kernel
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original to avoid over-sharpening
            result = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error in sharpening: {str(e)}")
            return image
    
    def correct_skew(self, image):
        """Correct skew in the image"""
        
        try:
            # Find skew angle using Hough line transform
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None:
                return image
            
            # Calculate average angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180 / np.pi
                # Filter angles to focus on text lines
                if 85 <= angle <= 95 or -5 <= angle <= 5:
                    angles.append(angle)
            
            if not angles:
                return image
            
            # Calculate correction angle
            avg_angle = np.mean(angles)
            if avg_angle > 45:
                skew_angle = avg_angle - 90
            else:
                skew_angle = avg_angle
            
            # Only correct if skew is significant
            if abs(skew_angle) > 0.5:
                # Get image center
                h, w = image.shape
                center = (w // 2, h // 2)
                
                # Create rotation matrix
                M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                
                # Apply rotation
                corrected = cv2.warpAffine(image, M, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
                return corrected
            
            return image
            
        except Exception as e:
            print(f"Error in skew correction: {str(e)}")
            return image
    
    def adaptive_binarization(self, image):
        """Apply adaptive binarization for better text extraction"""
        
        try:
            # Otsu's binarization
            _, binary1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Adaptive threshold
            binary2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            
            # Combine both methods
            combined = cv2.bitwise_and(binary1, binary2)
            
            return combined
            
        except Exception as e:
            print(f"Error in binarization: {str(e)}")
            # Fallback to simple threshold
            _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
            return binary
    
    def extract_text_with_confidence_regions(self, image_data, min_confidence=60):
        """Extract text with confidence mapping for quality assessment"""
        
        try:
            # Preprocess image
            processed_img = self.preprocess_image_advanced(image_data)
            
            # Get detailed OCR data with confidence
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:()'
            data = pytesseract.image_to_data(processed_img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter by confidence and build result
            high_confidence_text = []
            low_confidence_regions = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > min_confidence:
                    text = data['text'][i].strip()
                    if text:
                        high_confidence_text.append(text)
                else:
                    # Track low confidence regions for reprocessing
                    if data['text'][i].strip():
                        bbox = {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i],
                            'confidence': conf,
                            'text': data['text'][i]
                        }
                        low_confidence_regions.append(bbox)
            
            return {
                'text': ' '.join(high_confidence_text),
                'confidence_score': np.mean([int(c) for c in data['conf'] if int(c) > 0]),
                'low_confidence_regions': low_confidence_regions,
                'total_words': len([t for t in data['text'] if t.strip()]),
                'high_confidence_words': len(high_confidence_text)
            }
            
        except Exception as e:
            print(f"Error in confidence-based text extraction: {str(e)}")
            return {
                'text': '',
                'confidence_score': 0,
                'low_confidence_regions': [],
                'total_words': 0,
                'high_confidence_words': 0
            }
    
    def apply_specific_preprocessing(self, img_array, method):
        """Apply specific preprocessing method"""
        
        methods = {
            'gaussian_blur': lambda img: cv2.GaussianBlur(img, (3, 3), 0),
            'median_filter': lambda img: cv2.medianBlur(img, 3),
            'morphological_operations': self._morphological_ops,
            'contrast_enhancement': self.enhance_contrast,
            'noise_reduction': self.reduce_noise,
            'skew_correction': self.correct_skew
        }
        
        if method in methods:
            return methods[method](img_array)
        else:
            return img_array
    
    def _morphological_ops(self, image):
        """Apply morphological operations to clean up the image"""
        
        try:
            # Define kernel
            kernel = np.ones((2,2), np.uint8)
            
            # Apply morphological operations
            # Opening (erosion followed by dilation) - removes noise
            opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            
            # Closing (dilation followed by erosion) - fills small gaps
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
            
            return closed
            
        except Exception as e:
            print(f"Error in morphological operations: {str(e)}")
            return image