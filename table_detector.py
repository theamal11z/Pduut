import cv2
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
import io

class TableDetector:
    def __init__(self):
        self.min_table_area = 5000  # Minimum area to consider as table
        self.min_rows = 2
        self.min_cols = 2
    
    def detect_tables(self, image_data, page_number, assets_dir):
        """Detect and extract tables from page image"""
        
        tables = []
        
        try:
            # Convert image data to OpenCV format
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return tables
            
            # Detect table regions
            table_regions = self.find_table_regions(img)
            
            for i, region in enumerate(table_regions):
                try:
                    table_id = f"page_{page_number}_table_{i}"
                    
                    # Extract table data
                    table_data = self.extract_table_data(img, region, table_id, assets_dir)
                    
                    if table_data:
                        tables.append(table_data)
                
                except Exception as e:
                    print(f"Error processing table {i} on page {page_number}: {str(e)}")
                    continue
        
        except Exception as e:
            print(f"Error detecting tables on page {page_number}: {str(e)}")
        
        return tables
    
    def find_table_regions(self, img):
        """Find potential table regions using line detection"""
        
        regions = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Detect horizontal and vertical lines
            horizontal_lines = self.detect_lines(enhanced, horizontal=True)
            vertical_lines = self.detect_lines(enhanced, horizontal=False)
            
            # Create table mask by combining lines
            table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
            
            # Find contours representing potential tables
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by size and aspect ratio
                area = w * h
                if area >= self.min_table_area:
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 5:  # Reasonable table proportions
                        regions.append({
                            'bbox': (x, y, w, h),
                            'area': area
                        })
            
            # Sort by area (largest first)
            regions.sort(key=lambda r: r['area'], reverse=True)
            
        except Exception as e:
            print(f"Error finding table regions: {str(e)}")
        
        return regions[:5]  # Limit to top 5 potential tables
    
    def detect_lines(self, image, horizontal=True):
        """Detect horizontal or vertical lines in image"""
        
        try:
            # Apply threshold
            thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            # Create kernel for line detection
            if horizontal:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            # Detect lines
            lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            return lines
            
        except Exception as e:
            print(f"Error detecting lines: {str(e)}")
            return np.zeros_like(image)
    
    def extract_table_data(self, img, region, table_id, assets_dir):
        """Extract structured data from detected table region"""
        
        try:
            x, y, w, h = region['bbox']
            
            # Extract table region
            table_roi = img[y:y+h, x:x+w]
            
            # Save table image
            table_image_filename = f"{table_id}.png"
            table_image_path = os.path.join(assets_dir, "images", table_image_filename)
            cv2.imwrite(table_image_path, table_roi)
            
            # Attempt to extract table structure
            table_structure = self.analyze_table_structure(table_roi)
            
            # Create table data files
            csv_filename = f"{table_id}.csv"
            json_filename = f"{table_id}.json"
            
            csv_path = os.path.join(assets_dir, "tables", csv_filename)
            json_path = os.path.join(assets_dir, "tables", json_filename)
            
            # Generate sample structured data (in real implementation, this would use OCR on cells)
            table_data = self.extract_table_content(table_roi, table_structure)
            
            # Save as CSV
            if table_data.get('rows'):
                df = pd.DataFrame(table_data['rows'])
                df.to_csv(csv_path, index=False)
            
            # Save as JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(table_data, f, indent=2, ensure_ascii=False)
            
            return {
                "table_id": table_id,
                "csv_link": f"assets/tables/{csv_filename}",
                "json_link": f"assets/tables/{json_filename}",
                "image_link": f"assets/images/{table_image_filename}",
                "rows": table_structure.get('rows', 0),
                "cols": table_structure.get('cols', 0),
                "bbox": {"x": x, "y": y, "width": w, "height": h}
            }
            
        except Exception as e:
            print(f"Error extracting table data for {table_id}: {str(e)}")
            return None
    
    def analyze_table_structure(self, table_roi):
        """Analyze table structure to determine rows and columns"""
        
        structure = {"rows": 0, "cols": 0, "cells": []}
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal and vertical lines
            horizontal = self.detect_lines(gray, horizontal=True)
            vertical = self.detect_lines(gray, horizontal=False)
            
            # Find line positions
            horizontal_lines = self.find_line_positions(horizontal, horizontal=True)
            vertical_lines = self.find_line_positions(vertical, horizontal=False)
            
            # Calculate grid structure
            rows = max(1, len(horizontal_lines) - 1) if len(horizontal_lines) > 1 else self.estimate_rows(gray)
            cols = max(1, len(vertical_lines) - 1) if len(vertical_lines) > 1 else self.estimate_cols(gray)
            
            structure.update({
                "rows": rows,
                "cols": cols,
                "horizontal_lines": horizontal_lines,
                "vertical_lines": vertical_lines
            })
            
        except Exception as e:
            print(f"Error analyzing table structure: {str(e)}")
            # Fallback estimates
            h, w = table_roi.shape[:2]
            structure.update({
                "rows": max(2, h // 50),  # Rough estimate
                "cols": max(2, w // 100)  # Rough estimate
            })
        
        return structure
    
    def find_line_positions(self, line_image, horizontal=True):
        """Find positions of detected lines"""
        
        positions = []
        
        try:
            # Find contours of lines
            contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if horizontal:
                    # For horizontal lines, get y position
                    y = cv2.boundingRect(contour)[1]
                    positions.append(y)
                else:
                    # For vertical lines, get x position
                    x = cv2.boundingRect(contour)[0]
                    positions.append(x)
            
            # Sort and remove duplicates
            positions = sorted(list(set(positions)))
            
        except Exception as e:
            print(f"Error finding line positions: {str(e)}")
        
        return positions
    
    def estimate_rows(self, gray):
        """Estimate number of rows based on text density"""
        
        try:
            # Simple row estimation based on horizontal projection
            horizontal_projection = np.sum(gray < 128, axis=1)
            
            # Find peaks (text regions)
            peaks = []
            for i in range(1, len(horizontal_projection) - 1):
                if (horizontal_projection[i] > horizontal_projection[i-1] and 
                    horizontal_projection[i] > horizontal_projection[i+1] and
                    horizontal_projection[i] > np.mean(horizontal_projection)):
                    peaks.append(i)
            
            return max(self.min_rows, len(peaks))
            
        except Exception as e:
            print(f"Error estimating rows: {str(e)}")
            return self.min_rows
    
    def estimate_cols(self, gray):
        """Estimate number of columns based on text density"""
        
        try:
            # Simple column estimation based on vertical projection
            vertical_projection = np.sum(gray < 128, axis=0)
            
            # Find peaks (text regions)
            peaks = []
            for i in range(1, len(vertical_projection) - 1):
                if (vertical_projection[i] > vertical_projection[i-1] and 
                    vertical_projection[i] > vertical_projection[i+1] and
                    vertical_projection[i] > np.mean(vertical_projection)):
                    peaks.append(i)
            
            return max(self.min_cols, len(peaks))
            
        except Exception as e:
            print(f"Error estimating columns: {str(e)}")
            return self.min_cols
    
    def extract_table_content(self, table_roi, structure):
        """Extract content from table cells (simplified version)"""
        
        table_data = {
            "table_type": "detected_table",
            "rows": structure.get('rows', 0),
            "cols": structure.get('cols', 0),
            "extraction_method": "structure_analysis",
            "content": [],
            "raw_data": []
        }
        
        try:
            rows = structure.get('rows', 2)
            cols = structure.get('cols', 2)
            
            # Generate placeholder structure (in real implementation, would use OCR on each cell)
            for row in range(rows):
                row_data = []
                for col in range(cols):
                    # Placeholder cell content
                    cell_content = f"Cell({row+1},{col+1})"
                    row_data.append(cell_content)
                table_data["content"].append(row_data)
                table_data["raw_data"].append({
                    "row": row + 1,
                    "data": row_data
                })
            
        except Exception as e:
            print(f"Error extracting table content: {str(e)}")
        
        return table_data
