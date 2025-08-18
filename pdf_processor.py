# Robust import for PyMuPDF across versions / distributions
try:
    import pymupdf as fitz  # Preferred module name
except Exception:
    import fitz  # Fallback alias maintained by some versions
import os
import json
from datetime import datetime
from ocr_engine import OCREngine
from image_extractor import ImageExtractor
from table_detector import TableDetector
from equation_detector import EquationDetector
from utils import create_output_structure, sanitize_filename, ensure_directory
import io
from PIL import Image
import numpy as np
import cv2

class PDFProcessor:
    def __init__(self, ocr_languages=None, extract_images=True, extract_tables=True, extract_equations=True, use_advanced_ocr=False):
        self.ocr_languages = ocr_languages or ["eng"]
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_equations = extract_equations
        self.use_advanced_ocr = use_advanced_ocr
        
        # Initialize engines
        self.ocr_engine = OCREngine(languages=self.ocr_languages, use_advanced_preprocessing=use_advanced_ocr)
        self.image_extractor = ImageExtractor() if extract_images else None
        self.table_detector = TableDetector() if extract_tables else None
        self.equation_detector = EquationDetector() if extract_equations else None
    
    def process_pdf(self, pdf_path, output_dir, progress_callback=None):
        """Process a PDF file and extract structured data"""
        
        try:
            if progress_callback:
                progress_callback(15, "Opening PDF document...")
            
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Extract metadata
            metadata = self.extract_metadata(doc)
            
            if progress_callback:
                progress_callback(20, "Extracting PDF metadata...")
            
            # Create output structure
            assets_dir = os.path.join(output_dir, "assets")
            ensure_directory(os.path.join(assets_dir, "images"))
            ensure_directory(os.path.join(assets_dir, "tables"))
            ensure_directory(os.path.join(assets_dir, "text"))
            
            # Process pages
            pages_data = []
            total_pages = len(doc)
            
            for page_num in range(total_pages):
                if progress_callback:
                    progress = 20 + (page_num / total_pages) * 70
                    progress_callback(int(progress), f"Processing page {page_num + 1} of {total_pages}...")
                
                page_data = self.process_page(doc[page_num], page_num + 1, assets_dir)
                pages_data.append(page_data)
            
            doc.close()
            
            # Create final result
            result = create_output_structure(metadata, pages_data)
            
            if progress_callback:
                progress_callback(95, "Finalizing results...")
            
            return result
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def extract_metadata(self, doc):
        """Extract PDF metadata"""
        
        metadata = doc.metadata
        
        return {
            "title": metadata.get("title", "Unknown"),
            "author": metadata.get("author", "Unknown"),
            "subject": metadata.get("subject", "Educational Content"),
            "pages": len(doc),
            "creation_date": metadata.get("creationDate", ""),
            "modification_date": metadata.get("modDate", ""),
            "processed_date": datetime.now().isoformat()
        }
    
    def process_page(self, page, page_number, assets_dir):
        """Process a single page"""
        
        page_data = {
            "page_number": page_number,
            "text": "",
            "equations": [],
            "diagrams": [],
            "tables": [],
            "resources_links": {
                "page_text_link": "",
                "equations_links": [],
                "diagrams_links": [],
                "tables_links": []
            }
        }
        
        try:
            # Get page as image for OCR and analysis (force RGB, no alpha)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(3, 3),  # bump to 3x for better OCR/equations
                colorspace=getattr(fitz, 'csRGB', None),
                alpha=False
            )
            img_data = pix.tobytes("png")
            
            # Fallback: if rendered image is too dark (black), re-render with alpha and composite over white
            try:
                nparr_chk = np.frombuffer(img_data, np.uint8)
                img_chk = cv2.imdecode(nparr_chk, cv2.IMREAD_COLOR)
                if img_chk is None or float(img_chk.mean()) < 10.0:
                    pix_a = page.get_pixmap(
                        matrix=fitz.Matrix(3, 3),
                        colorspace=getattr(fitz, 'csRGB', None),
                        alpha=True
                    )
                    rgba = Image.open(io.BytesIO(pix_a.tobytes("png"))).convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.split()[-1])
                    buf = io.BytesIO()
                    bg.save(buf, format="PNG")
                    img_data = buf.getvalue()
            except Exception:
                pass
            
            # Optional preprocessing to improve OCR & detectors (CLAHE + unsharp)
            try:
                nparr = np.frombuffer(img_data, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l2 = clahe.apply(l)
                    lab2 = cv2.merge((l2, a, b))
                    img_eq = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
                    # mild denoise and sharpen
                    blur = cv2.GaussianBlur(img_eq, (0, 0), sigmaX=1.0)
                    img_sharp = cv2.addWeighted(img_eq, 1.5, blur, -0.5, 0)
                    # encode back to PNG bytes
                    ok, enc = cv2.imencode('.png', img_sharp)
                    if ok:
                        img_data = enc.tobytes()
            except Exception:
                pass

            # Extract text - combine native PDF text and OCR for complete coverage
            extracted_text = ""
            
            # First try to extract native text from PDF
            try:
                native_text = page.get_text()
                if native_text and native_text.strip():
                    extracted_text = native_text
            except:
                pass
            
            # If no native text or text is minimal, use OCR
            if self.ocr_engine and (not extracted_text or len(extracted_text.strip()) < 100):
                ocr_text = self.ocr_engine.extract_text_from_image_data(img_data)
                if ocr_text and len(ocr_text.strip()) > len(extracted_text.strip()):
                    extracted_text = ocr_text
                elif extracted_text and ocr_text:
                    # Combine both if they contain different content
                    combined_text = f"{extracted_text}\n\n--- OCR Content ---\n{ocr_text}"
                    extracted_text = combined_text
            
            # For very short text, always add OCR to ensure complete coverage
            if self.ocr_engine and len(extracted_text.strip()) < 200:
                ocr_text = self.ocr_engine.extract_text_from_image_data(img_data)
                if ocr_text and ocr_text.strip() not in extracted_text:
                    if extracted_text.strip():
                        extracted_text = f"{extracted_text}\n\n--- Additional OCR Content ---\n{ocr_text}"
                    else:
                        extracted_text = ocr_text
            
            page_data["text"] = extracted_text
            
            # Save text to file
            text_filename = f"page_{page_number}.txt"
            text_path = os.path.join(assets_dir, "text", text_filename)
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(page_data["text"])
            page_data["resources_links"]["page_text_link"] = f"assets/text/{text_filename}"
            
            # Extract equations
            if self.extract_equations and self.equation_detector:
                equations = self.equation_detector.detect_equations(page_data["text"], img_data, page_number, assets_dir=assets_dir)
                page_data["equations"] = equations
                page_data["resources_links"]["equations_links"] = [eq.get("image_link", "") for eq in equations]
            
            # Extract images/diagrams
            if self.extract_images and self.image_extractor:
                diagrams = self.image_extractor.extract_images_from_page(page, page_number, assets_dir)
                page_data["diagrams"] = diagrams
                page_data["resources_links"]["diagrams_links"] = [diag.get("image_link", "") for diag in diagrams]
            
            # Detect tables
            if self.extract_tables and self.table_detector:
                tables = self.table_detector.detect_tables(img_data, page_number, assets_dir)
                page_data["tables"] = tables
                page_data["resources_links"]["tables_links"] = [table.get("json_link", "") for table in tables]
            
        except Exception as e:
            print(f"Error processing page {page_number}: {str(e)}")
        
        return page_data
