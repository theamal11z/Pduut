import fitz  # PyMuPDF
import os
import json
from datetime import datetime
from ocr_engine import OCREngine
from image_extractor import ImageExtractor
from table_detector import TableDetector
from equation_detector import EquationDetector
from utils import create_output_structure, sanitize_filename, ensure_directory

class PDFProcessor:
    def __init__(self, ocr_languages=None, extract_images=True, extract_tables=True, extract_equations=True):
        self.ocr_languages = ocr_languages or ["eng"]
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.extract_equations = extract_equations
        
        # Initialize engines
        self.ocr_engine = OCREngine(languages=self.ocr_languages)
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
            # Get page as image for OCR and analysis
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
            img_data = pix.tobytes("png")
            
            # Extract text using OCR
            if self.ocr_engine:
                page_data["text"] = self.ocr_engine.extract_text_from_image_data(img_data)
                
                # Save text to file
                text_filename = f"page_{page_number}.txt"
                text_path = os.path.join(assets_dir, "text", text_filename)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(page_data["text"])
                page_data["resources_links"]["page_text_link"] = f"assets/text/{text_filename}"
            
            # Extract equations
            if self.extract_equations and self.equation_detector:
                equations = self.equation_detector.detect_equations(page_data["text"], img_data, page_number)
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
