import fitz
import cv2
import numpy as np
from PIL import Image
import os
import json

class ImageExtractor:
    def __init__(self):
        self.min_image_size = (50, 50)  # Minimum size to consider as meaningful image
        self.max_images_per_page = 20   # Prevent extraction of too many small artifacts
    
    def extract_images_from_page(self, page, page_number, assets_dir):
        """Extract images and diagrams from a PDF page"""
        
        diagrams = []
        
        try:
            # Method 1: Extract embedded images
            embedded_images = self.extract_embedded_images(page, page_number, assets_dir)
            diagrams.extend(embedded_images)
            
            # Method 2: Detect image regions using computer vision
            cv_images = self.detect_image_regions(page, page_number, assets_dir)
            diagrams.extend(cv_images)
            
            # Limit number of images
            if len(diagrams) > self.max_images_per_page:
                diagrams = diagrams[:self.max_images_per_page]
            
        except Exception as e:
            print(f"Error extracting images from page {page_number}: {str(e)}")
        
        return diagrams
    
    def extract_embedded_images(self, page, page_number, assets_dir):
        """Extract embedded images from PDF page"""
        
        diagrams = []
        
        try:
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Create unique ID
                    diagram_id = f"page_{page_number}_embedded_{img_index}"
                    
                    # Save image
                    image_filename = f"{diagram_id}.png"
                    image_path = os.path.join(assets_dir, "images", image_filename)
                    
                    # Convert to PIL Image and save as PNG
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    # Check minimum size
                    if image.size[0] >= self.min_image_size[0] and image.size[1] >= self.min_image_size[1]:
                        image.save(image_path, "PNG")
                        
                        # Generate description
                        description = self.analyze_image_content(image)
                        
                        diagram_data = {
                            "diagram_id": diagram_id,
                            "description": description,
                            "image_link": f"assets/images/{image_filename}",
                            "width": image.size[0],
                            "height": image.size[1],
                            "extraction_method": "embedded"
                        }
                        
                        diagrams.append(diagram_data)
                
                except Exception as e:
                    print(f"Error processing embedded image {img_index}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error extracting embedded images: {str(e)}")
        
        return diagrams
    
    def detect_image_regions(self, page, page_number, assets_dir):
        """Detect image regions using computer vision techniques"""
        
        diagrams = []
        
        try:
            # Get page as high-resolution image
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
            img_data = pix.tobytes("png")
            
            # Convert to OpenCV format
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return diagrams
            
            # Detect potential diagram regions
            regions = self.find_diagram_regions(img)
            
            for i, region in enumerate(regions):
                try:
                    x, y, w, h = region
                    
                    # Extract region
                    roi = img[y:y+h, x:x+w]
                    
                    if roi.size == 0:
                        continue
                    
                    # Create unique ID
                    diagram_id = f"page_{page_number}_region_{i}"
                    
                    # Save region as image
                    image_filename = f"{diagram_id}.png"
                    image_path = os.path.join(assets_dir, "images", image_filename)
                    cv2.imwrite(image_path, roi)
                    
                    # Generate description
                    description = self.analyze_region_content(roi)
                    
                    diagram_data = {
                        "diagram_id": diagram_id,
                        "description": description,
                        "image_link": f"assets/images/{image_filename}",
                        "width": w,
                        "height": h,
                        "extraction_method": "region_detection",
                        "bbox": {"x": x, "y": y, "width": w, "height": h}
                    }
                    
                    diagrams.append(diagram_data)
                
                except Exception as e:
                    print(f"Error processing region {i}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error in region detection: {str(e)}")
        
        return diagrams
    
    def find_diagram_regions(self, img):
        """Find potential diagram/image regions using computer vision"""
        
        regions = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size
                if w >= self.min_image_size[0] and h >= self.min_image_size[1]:
                    # Check aspect ratio (avoid very thin/wide rectangles)
                    aspect_ratio = w / h
                    if 0.1 <= aspect_ratio <= 10:
                        # Check area
                        area = w * h
                        if area >= 2500:  # Minimum area threshold
                            regions.append((x, y, w, h))
            
            # Sort by area (largest first)
            regions.sort(key=lambda r: r[2] * r[3], reverse=True)
            
            # Remove overlapping regions
            filtered_regions = self.remove_overlapping_regions(regions)
            
        except Exception as e:
            print(f"Error finding diagram regions: {str(e)}")
        
        return filtered_regions[:10]  # Limit to top 10 regions
    
    def remove_overlapping_regions(self, regions, overlap_threshold=0.7):
        """Remove overlapping regions"""
        
        if not regions:
            return regions
        
        filtered = []
        
        for i, region1 in enumerate(regions):
            is_overlapping = False
            
            for j, region2 in enumerate(filtered):
                if self.calculate_overlap(region1, region2) > overlap_threshold:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                filtered.append(region1)
        
        return filtered
    
    def calculate_overlap(self, region1, region2):
        """Calculate overlap ratio between two regions"""
        
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        # Calculate intersection
        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1 + w1, x2 + w2) - xi)
        hi = max(0, min(y1 + h1, y2 + h2) - yi)
        
        intersection = wi * hi
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def analyze_image_content(self, image):
        """Analyze image content to generate description"""
        
        try:
            # Basic analysis based on image properties
            width, height = image.size
            aspect_ratio = width / height
            
            # Determine likely content type
            if aspect_ratio > 2:
                content_type = "wide diagram or chart"
            elif aspect_ratio < 0.5:
                content_type = "tall diagram or figure"
            else:
                content_type = "diagram or illustration"
            
            # Check if image is mostly text (simple heuristic)
            gray = image.convert('L')
            pixels = np.array(gray)
            white_ratio = np.sum(pixels > 200) / pixels.size
            
            if white_ratio > 0.8:
                content_type = "text-heavy figure or table"
            
            return f"Educational {content_type} ({width}x{height}px)"
            
        except Exception as e:
            print(f"Error analyzing image content: {str(e)}")
            return "Educational diagram or illustration"
    
    def analyze_region_content(self, roi):
        """Analyze detected region content"""
        
        try:
            h, w = roi.shape[:2]
            
            # Simple content analysis
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Check edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size
            
            if edge_ratio > 0.1:
                content_type = "complex diagram with multiple elements"
            elif edge_ratio > 0.05:
                content_type = "diagram or chart"
            else:
                content_type = "simple figure or text block"
            
            return f"Detected {content_type} ({w}x{h}px)"
            
        except Exception as e:
            print(f"Error analyzing region content: {str(e)}")
            return f"Detected visual element ({roi.shape[1]}x{roi.shape[0]}px)"

import io
