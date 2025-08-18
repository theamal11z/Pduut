import re
import cv2
import numpy as np
import json
import os
from PIL import Image
import io
import pytesseract

class EquationDetector:
    def __init__(self):
        # Mathematical symbols and patterns
        self.math_patterns = [
            r'\b\d+\s*[+\-*/=]\s*\d+',  # Basic arithmetic
            r'[a-zA-Z]\s*[+\-*/=]\s*[a-zA-Z0-9]',  # Algebraic expressions
            r'\b\d*[a-zA-Z]\^?\d*',  # Variables with coefficients
            r'[∫∑∏√±∞≤≥≠≈∂∇]',  # Mathematical symbols
            r'\b(sin|cos|tan|log|ln|exp|sqrt)\s*\(',  # Functions
            r'\b\d+\.\d+',  # Decimals
            r'\([^)]*[+\-*/=][^)]*\)',  # Expressions in parentheses
            r'[a-zA-Z]\s*=\s*[^,\n.;]+',  # Equations
            r'\b(dx|dy|dt|dr)\b',  # Calculus notation
            r'[αβγδεζηθικλμνξοπρστυφχψω]',  # Greek letters
        ]
        
        # Compile patterns
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.math_patterns]
    
    def detect_equations(self, text, image_data, page_number, assets_dir=None):
        """Detect mathematical equations in text and image.
        If assets_dir is provided, save cropped equation images to assets/images.
        """
        
        equations = []
        
        try:
            # Text-based detection
            text_equations = self.detect_equations_in_text(text, page_number)
            equations.extend(text_equations)
            
            # Image-based detection (simplified)
            image_equations = self.detect_equations_in_image(image_data, page_number, assets_dir=assets_dir)
            equations.extend(image_equations)
            
        except Exception as e:
            print(f"Error detecting equations on page {page_number}: {str(e)}")
        
        return equations
    
    def detect_equations_in_text(self, text, page_number):
        """Detect equations in extracted text"""
        
        equations = []
        
        if not text:
            return equations
        
        try:
            # Split text into lines
            lines = text.split('\n')
            
            for line_idx, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check each pattern
                for pattern_idx, pattern in enumerate(self.compiled_patterns):
                    matches = pattern.finditer(line)
                    
                    for match_idx, match in enumerate(matches):
                        equation_text = match.group()
                        
                        # Skip if too short or too common
                        if len(equation_text) < 3 or self.is_common_word(equation_text):
                            continue
                        
                        # Create equation ID
                        equation_id = f"page_{page_number}_text_{line_idx}_{pattern_idx}_{match_idx}"
                        
                        # Convert to LaTeX (simplified)
                        latex = self.text_to_latex(equation_text)
                        
                        equation_data = {
                            "equation_id": equation_id,
                            "latex": latex,
                            "original_text": equation_text,
                            "detection_method": "text_pattern",
                            "confidence": self.calculate_confidence(equation_text),
                            "line_number": line_idx + 1,
                            "image_link": ""  # No image for text-based detection
                        }
                        
                        equations.append(equation_data)
            
        except Exception as e:
            print(f"Error detecting equations in text: {str(e)}")
        
        return equations
    
    def detect_equations_in_image(self, image_data, page_number, assets_dir=None):
        """Detect mathematical equations in image using visual patterns.
        Saves crops when assets_dir is provided.
        """
        
        equations = []
        
        try:
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return equations
            
            # Find potential equation regions (enhanced)
            equation_regions = self.find_equation_regions(img)
            
            for i, region in enumerate(equation_regions):
                try:
                    equation_id = f"page_{page_number}_visual_{i}"
                    
                    # Extract equation region
                    x, y, w, h = region['bbox']
                    equation_roi = img[y:y+h, x:x+w]
                    
                    # Save equation image if path provided
                    equation_filename = f"{equation_id}.png"
                    if assets_dir:
                        import os
                        images_dir = os.path.join(assets_dir, "images")
                        os.makedirs(images_dir, exist_ok=True)
                        _ = cv2.imwrite(os.path.join(images_dir, equation_filename), equation_roi)
                    
                    # Analyze equation content
                    latex = self.image_to_latex(equation_roi)
                    
                    equation_data = {
                        "equation_id": equation_id,
                        "latex": latex,
                        "original_text": "",
                        "detection_method": "visual_pattern",
                        "confidence": region.get('confidence', 0.5),
                        "bbox": {"x": x, "y": y, "width": w, "height": h},
                        "image_link": f"assets/images/{equation_filename}" if assets_dir else ""
                    }
                    
                    equations.append(equation_data)
                
                except Exception as e:
                    print(f"Error processing equation region {i}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error detecting equations in image: {str(e)}")
        
        return equations
    
    def find_equation_regions(self, img):
        """Find regions likely to contain mathematical equations using thresholding,
        morphology, and MSER heuristics.
        """
        
        regions = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Adaptive threshold to emphasize text/equations
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 35, 10)

            # Morphological close to join characters into lines/blocks
            kx = max(15, img.shape[1] // 80)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
            closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)

            # MSER to detect stable text-like regions
            try:
                mser = cv2.MSER_create(_min_area=60, _max_area=10000)
                regions, _ = mser.detectRegions(gray)
                mser_mask = np.zeros_like(gray, dtype=np.uint8)
                for pts in regions:
                    cv2.fillPoly(mser_mask, [pts.reshape(-1, 1, 2)], 255)
                fused = cv2.bitwise_or(closed, mser_mask)
            except Exception:
                fused = closed

            # Find contours as candidate blocks
            contours, _ = cv2.findContours(fused, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            equation_candidates = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if w < 40 or h < 12:
                    continue
                ratio = w / float(h)
                area = w * h
                # Equations tend to be wider than tall or compact mathematical blocks
                if ratio > 1.4 and area > 800:
                    roi = gray[y:y+h, x:x+w]
                    symbol_count = self.count_math_features(roi)
                    equation_candidates.append({
                        'bbox': (x, y, w, h),
                        'symbol_count': symbol_count,
                        'area': area
                    })
            
            for candidate in equation_candidates:
                # Simple confidence scoring based on symbol density
                confidence = min(1.0, candidate.get('symbol_count', 1) * 0.2)
                
                if confidence > 0.3:  # Minimum confidence threshold
                    regions.append({
                        'bbox': candidate['bbox'],
                        'confidence': confidence,
                        'symbol_count': candidate.get('symbol_count', 1)
                    })
            
            # Sort by confidence
            regions.sort(key=lambda r: r['confidence'], reverse=True)
            
        except Exception as e:
            print(f"Error finding equation regions: {str(e)}")
        
        return regions[:10]  # Limit to top 10 candidates
    
    def detect_math_symbols(self, gray_image):
        """Detect mathematical symbols in grayscale image"""
        
        candidates = []
        
        try:
            # Apply edge detection
            edges = cv2.Canny(gray_image, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size (equations are usually longer than tall)
                if w > 50 and h > 10 and w/h > 1.5:
                    area = w * h
                    
                    # Check for mathematical characteristics
                    roi = gray_image[y:y+h, x:x+w]
                    symbol_count = self.count_math_features(roi)
                    
                    if symbol_count > 0:
                        candidates.append({
                            'bbox': (x, y, w, h),
                            'symbol_count': symbol_count,
                            'area': area
                        })
        
        except Exception as e:
            print(f"Error detecting math symbols: {str(e)}")
        
        return candidates
    
    def count_math_features(self, roi):
        """Count mathematical features in a region of interest"""
        
        feature_count = 0
        
        try:
            # Simple heuristics for mathematical content
            
            # Check for horizontal lines (fraction bars, equals signs)
            horizontal_lines = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            horizontal = cv2.morphologyEx(roi, cv2.MORPH_OPEN, horizontal_lines)
            if np.sum(horizontal > 0) > 100:
                feature_count += 1
            
            # Check for vertical arrangements (superscripts, subscripts)
            height_variance = np.var(np.sum(roi < 128, axis=1))
            if height_variance > 1000:
                feature_count += 1
            
            # Check for special shapes (circles, squares for operators)
            circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=25)
            if circles is not None:
                feature_count += len(circles[0])
            
        except Exception as e:
            print(f"Error counting math features: {str(e)}")
        
        return feature_count
    
    def text_to_latex(self, text):
        """Convert mathematical text to LaTeX format (simplified)"""
        
        latex = text
        
        try:
            # Basic conversions
            conversions = {
                '*': '\\cdot ',
                '+-': '\\pm ',
                '>=': '\\geq ',
                '<=': '\\leq ',
                '!=': '\\neq ',
                '~=': '\\approx ',
                'sqrt': '\\sqrt',
                'sum': '\\sum',
                'integral': '\\int',
                'infinity': '\\infty',
                'alpha': '\\alpha',
                'beta': '\\beta',
                'gamma': '\\gamma',
                'delta': '\\delta',
                'pi': '\\pi',
                'theta': '\\theta'
            }
            
            for text_form, latex_form in conversions.items():
                latex = latex.replace(text_form, latex_form)
            
            # Handle exponents (simple case)
            latex = re.sub(r'([a-zA-Z0-9])\^([a-zA-Z0-9]+)', r'\1^{\2}', latex)
            
            # Handle fractions (simple case)
            latex = re.sub(r'([a-zA-Z0-9]+)/([a-zA-Z0-9]+)', r'\\frac{\1}{\2}', latex)
            
            # Wrap in math mode if not already
            if not latex.startswith('$'):
                latex = f"${latex}$"
            
        except Exception as e:
            print(f"Error converting text to LaTeX: {str(e)}")
            latex = f"${text}$"  # Fallback
        
        return latex
    
    def image_to_latex(self, equation_image):
        """Convert equation image to LaTeX-like string using OCR heuristics.
        Not a perfect converter; aims to avoid generic placeholders.
        """
        try:
            # Preprocess ROI for OCR
            if len(equation_image.shape) == 3:
                gray = cv2.cvtColor(equation_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = equation_image

            gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 31, 10)
            # Remove small noise
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            proc = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

            # OCR with math-friendly whitelist
            config = (
                "--psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+\-*/=^()[]{}.,;:_\\ "+
                "-c preserve_interword_spaces=1"
            )
            text = pytesseract.image_to_string(proc, config=config)
            text = text.strip()

            if not text:
                return "$\\text{Equation}$"

            # Basic normalization
            text = re.sub(r"\s+", " ", text)
            # Simple LaTeX-ish conversions
            replacements = {
                "*": " \\cdot ",
                "+-": " \\pm ",
                ">=": " \\geq ",
                "<=": " \\leq ",
                "!=": " \\neq ",
                "~=": " \\approx ",
            }
            for k, v in replacements.items():
                text = text.replace(k, v)

            # Exponents a^b -> a^{b}
            text = re.sub(r"([A-Za-z0-9])\^([A-Za-z0-9]+)", r"\1^{\2}", text)
            # Fractions a/b -> \\frac{a}{b} when simple
            text = re.sub(r"\b([A-Za-z0-9]+)\s*/\s*([A-Za-z0-9]+)\b", r"\\frac{\1}{\2}", text)

            # Limit very long strings
            if len(text) > 120:
                text = text[:117] + "..."

            # Wrap in math mode
            if not text.startswith("$"):
                text = f"${text}$"
            return text
        except Exception as e:
            print(f"Error converting image to LaTeX: {str(e)}")
            return "$\\text{Equation}$"
    
    def calculate_confidence(self, equation_text):
        """Calculate confidence score for detected equation"""
        
        confidence = 0.1  # Base confidence
        
        try:
            # Length factor
            if len(equation_text) > 5:
                confidence += 0.2
            
            # Mathematical operators
            math_ops = ['+', '-', '*', '/', '=', '^', '(', ')']
            op_count = sum(equation_text.count(op) for op in math_ops)
            confidence += min(0.3, op_count * 0.1)
            
            # Numbers
            digit_count = sum(c.isdigit() for c in equation_text)
            confidence += min(0.2, digit_count * 0.05)
            
            # Mathematical functions
            functions = ['sin', 'cos', 'tan', 'log', 'sqrt', 'exp']
            for func in functions:
                if func in equation_text.lower():
                    confidence += 0.15
                    break
            
            # Greek letters or special symbols
            special_chars = ['α', 'β', 'γ', 'δ', 'π', 'θ', '∫', '∑', '√', '∞']
            for char in special_chars:
                if char in equation_text:
                    confidence += 0.1
                    break
            
        except Exception as e:
            print(f"Error calculating confidence: {str(e)}")
        
        return min(1.0, confidence)
    
    def is_common_word(self, text):
        """Check if text is a common word (not mathematical)"""
        
        common_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'through', 'during'
        }
        
        return text.lower().strip() in common_words
