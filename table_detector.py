import cv2
import numpy as np
import pandas as pd
import json
import os
from PIL import Image
import io
import pytesseract

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
            
            # Save as CSV using the content data
            try:
                if table_data.get('content') and isinstance(table_data['content'], list) and len(table_data['content']) > 0:
                    # Use the structured content for CSV
                    df = pd.DataFrame(table_data['content'])
                    df.to_csv(csv_path, index=False, header=False)
                else:
                    # Create a basic CSV file with table information
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write("Table_ID,Rows,Columns,Detection_Method\n")
                        f.write(f"{table_id},{table_data.get('rows', 0)},{table_data.get('cols', 0)},structure_analysis\n")
            except Exception as e:
                print(f"Error creating CSV for {table_id}: {str(e)}")
                # Create a minimal CSV file
                with open(csv_path, 'w', encoding='utf-8') as f:
                    f.write("Table_Info\n")
                    f.write(f"Table detected: {table_id}\n")
                    f.write(f"Rows: {table_data.get('rows', 'Unknown')}\n")
                    f.write(f"Columns: {table_data.get('cols', 'Unknown')}\n")
            
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
        """Extract content from table cells using OCR.
        Uses detected grid lines when available; otherwise splits uniformly.
        """
        
        table_data = {
            "table_type": "detected_table",
            "rows": structure.get('rows', 0),
            "cols": structure.get('cols', 0),
            "extraction_method": "ocr_cells",
            "content": [],
            "cell_data": []
        }
        
        try:
            rows = max(1, int(structure.get('rows', 1)))
            cols = max(1, int(structure.get('cols', 1)))

            # Preprocess for OCR
            gray = cv2.cvtColor(table_roi, cv2.COLOR_BGR2GRAY)
            thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 31, 10)

            h, w = gray.shape[:2]
            h_lines = structure.get('horizontal_lines') or []
            v_lines = structure.get('vertical_lines') or []

            # Build grid boundaries
            if len(h_lines) >= 2 and len(v_lines) >= 2:
                ys = sorted(h_lines)
                xs = sorted(v_lines)
            else:
                # Uniform splits
                ys = [int(i * h / rows) for i in range(rows + 1)]
                xs = [int(j * w / cols) for j in range(cols + 1)]

            # Ensure bounds cover full image
            ys[0] = 0
            ys[-1] = h
            xs[0] = 0
            xs[-1] = w

            ocr_config = "--psm 6"

            for r in range(rows):
                row_data = []
                y0 = ys[r] if r < len(ys) - 1 else int(r * h / rows)
                y1 = ys[r + 1] if r + 1 < len(ys) else int((r + 1) * h / rows)
                y0, y1 = max(0, y0), min(h, y1)
                for c in range(cols):
                    x0 = xs[c] if c < len(xs) - 1 else int(c * w / cols)
                    x1 = xs[c + 1] if c + 1 < len(xs) else int((c + 1) * w / cols)
                    x0, x1 = max(0, x0), min(w, x1)
                    if y1 <= y0 or x1 <= x0:
                        row_data.append("")
                        continue

                    cell = thr[y0:y1, x0:x1]

                    # Small border to reduce line noise
                    pad = max(1, min(cell.shape[0], cell.shape[1]) // 50)
                    cell_clean = cell[pad:-pad, pad:-pad] if cell.shape[0] > 2*pad and cell.shape[1] > 2*pad else cell

                    try:
                        text = pytesseract.image_to_string(cell_clean, config=ocr_config)
                        text = text.strip()
                    except Exception:
                        text = ""

                    row_data.append(text)
                table_data["content"].append(row_data)

            table_data["cell_data"] = {
                "total_cells": rows * cols,
                "grid_size": f"{rows}x{cols}",
                "detection_confidence": "medium"
            }
        except Exception as e:
            print(f"Error extracting table content: {str(e)}")
        
        return table_data
